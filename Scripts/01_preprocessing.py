"""Individual EEG preprocessing and collection."""

import traceback
import numpy as np
import pandas as pd
import mne
import autoreject
from joblib import Parallel, delayed
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA, annotate_amplitude
from mne_icalabel import label_components
from pyprep.find_noisy_channels import NoisyChannels
import matplotlib.pyplot as plt
from config import N_JOBS, OVERWRITE, PREPROCESS_DIR, basepath, derivpath


def main():
    GROUPS = {
        "ncbp": {
            "display_name": "NCBP",
            "data_folder": "raw_data_ncbp",
            "task": "rest",
            "session": "pre",
            "notch_hz": 60,
            "crop_mode": "fixed",
        },
        "ncbphc": {
            "display_name": "NCBP-HC",
            "data_folder": "raw_data_ncbpmc",
            "task": "restingclose",
            "session": None,
            "notch_hz": 60,
            "crop_mode": "fixed",
            "additional_drop_channels": ["rating", "stim", "thermode"],
        },
    }
    COMMON_DROP_CHANNELS = [
        "HEOG",
        "VEOG",
        "HEOGL",
        "ECG",
        "GSR",
        "HEOGR",
        "VEOGL",
        "VEOGR",
        "GSR",
        "ECG",
        "LE",
        "RE",
        "Ne",
        "Ma",
        "DIN1",
        "RSTR",
        "STRT",
        "t001",
        "t002",
        "t003",
        "t004",
        "t005",
        "END+",
    ]
    PROCESS_COLUMNS = [
        "n_bad_chans",
        "n_removed_icas_total",
        "ica_muscle_artifact",
        "ica_eye_blink",
        "ica_heart_beat",
        "ica_channel_noise",
        "ica_line_noise",
        "n_interp_bads",
        "n_amplitude_bad_segments",
        "n_amplitude_bad_channels",
        "n_epochs_before_autoreject",
        "n_epochs_after_autoreject",
        "n_epochs_dropped_by_annotation",
        "n_epochs_autoreject_dropped",
        "percent_epochs_kept",
    ]

    def crop_recording(raw, group_info):
        if group_info["crop_mode"] == "fixed":
            start_sec = 0.0
        else:
            descriptions = np.asarray(raw.annotations.description, dtype=str)
            trigger_idx = np.where(descriptions == group_info["crop_trigger"])[0][0]
            start_sec = float(raw.annotations.onset[trigger_idx])
        available_duration = raw.times[-1] - start_sec
        if available_duration > 300:
            end_sec = start_sec + 300
        else:
            end_sec = raw.times[-1]
        return raw.copy().crop(tmin=start_sec, tmax=end_sec)

    def all_subjects_have_iz(sub_ids, group_info, datapath):
        for sub_id in sub_ids:
            bids_path = BIDSPath(
                subject=sub_id.replace("sub-", ""),
                task=group_info["task"],
                session=group_info["session"],
                datatype="eeg",
                root=datapath,
            )
            raw = read_raw_bids(bids_path=bids_path, verbose="ERROR")
            if "Iz" not in raw.ch_names:
                return False
        return True

    def preprocess_subject(sub_id, group_key, group_info, datapath, derivpath, keep_iz):
        sub_out = derivpath / sub_id
        sub_out.mkdir(parents=True, exist_ok=True)
        bids_path = BIDSPath(
            subject=sub_id.replace("sub-", ""),
            task=group_info["task"],
            session=group_info["session"],
            datatype="eeg",
            root=datapath,
        )
        raw = read_raw_bids(bids_path=bids_path, verbose="ERROR")
        raw = crop_recording(raw, group_info)
        drop_channels = COMMON_DROP_CHANNELS + group_info.get(
            "additional_drop_channels", []
        )
        if not keep_iz:
            drop_channels.append("Iz")
        drop_channels = [ch for ch in drop_channels if ch in raw.ch_names]
        raw.drop_channels(drop_channels, on_missing="ignore")
        raw.pick("eeg")
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="warn")
        raw.load_data()
        raw.notch_filter(freqs=group_info["notch_hz"], verbose="ERROR")
        raw.filter(l_freq=1, h_freq=100, verbose="ERROR")
        raw.resample(250, verbose="ERROR")
        noisy_detector = NoisyChannels(raw)
        noisy_detector.find_all_bads(ransac=True)
        raw.info["bads"] = sorted(set(raw.info["bads"] + noisy_detector.get_bads()))
        if "FCz" not in raw.ch_names:
            raw = mne.add_reference_channels(raw, ref_channels=["FCz"], copy=False)
        raw.set_montage(montage, on_missing="warn")
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
        stats = {column: np.nan for column in PROCESS_COLUMNS}
        stats["n_bad_chans"] = len(raw.info["bads"])
        ica_events = mne.make_fixed_length_events(raw, id=1, duration=5)
        ica_epochs = mne.Epochs(
            raw,
            ica_events,
            tmin=0,
            tmax=5 - 1 / raw.info["sfreq"],
            baseline=None,
            reject=dict(eeg=300e-6),
            flat=dict(eeg=1e-06),
            preload=True,
            verbose=False,
        )
        ica = ICA(method="infomax", fit_params=dict(extended=True), random_state=42)
        ica_picks = mne.pick_types(ica_epochs.info, eeg=True, exclude="bads")
        ica.fit(ica_epochs, picks=ica_picks, verbose="ERROR")
        ica_labels = label_components(ica_epochs, ica, method="iclabel")
        ica_df = pd.DataFrame(
            {
                "component": np.arange(len(ica_labels["labels"])),
                "label": ica_labels["labels"],
                "probability": ica_labels["y_pred_proba"],
            }
        )
        artifact_classes = {
            "muscle artifact",
            "eye blink",
            "heart beat",
            "channel noise",
            "line noise",
        }
        ica_df["rejected"] = ica_df["label"].isin(artifact_classes) & (
            ica_df["probability"] > 0.7
        )
        ica.exclude = ica_df.loc[ica_df["rejected"], "component"].tolist()
        stats["n_removed_icas_total"] = int(ica_df["rejected"].sum())
        stats["ica_muscle_artifact"] = int(
            ((ica_df["label"] == "muscle artifact") & ica_df["rejected"]).sum()
        )
        stats["ica_eye_blink"] = int(
            ((ica_df["label"] == "eye blink") & ica_df["rejected"]).sum()
        )
        stats["ica_heart_beat"] = int(
            ((ica_df["label"] == "heart beat") & ica_df["rejected"]).sum()
        )
        stats["ica_channel_noise"] = int(
            ((ica_df["label"] == "channel noise") & ica_df["rejected"]).sum()
        )
        stats["ica_line_noise"] = int(
            ((ica_df["label"] == "line noise") & ica_df["rejected"]).sum()
        )
        ica.apply(raw, verbose="ERROR")
        stats["n_interp_bads"] = len(raw.info["bads"])
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        bad_annotations, amplitude_bad_channels = annotate_amplitude(
            raw,
            peak=dict(eeg=100e-6),
            flat=None,
            bad_percent=5,
            min_duration=0.005,
            picks="eeg",
            verbose="ERROR",
        )
        raw.set_annotations(raw.annotations + bad_annotations)
        stats["n_amplitude_bad_segments"] = len(bad_annotations)
        stats["n_amplitude_bad_channels"] = len(amplitude_bad_channels)
        raw_psd = raw.compute_psd(
            method="welch",
            fmin=1,
            fmax=100,
            picks="eeg",
            exclude="bads",
            reject_by_annotation=True,
            n_jobs=1,
            verbose="ERROR",
        )
        psd_fig = raw_psd.plot(show=False)
        psd_fig.savefig(sub_out / f"{sub_id}_psd.svg", dpi=600)
        plt.close(psd_fig)
        events = mne.make_fixed_length_events(raw, id=1, duration=5, overlap=2.5)
        epochs = mne.Epochs(
            raw,
            events,
            tmin=0,
            tmax=5 - 1 / raw.info["sfreq"],
            baseline=None,
            reject=None,
            flat=None,
            reject_by_annotation=True,
            preload=True,
            verbose=False,
        )
        drop_fig = mne.viz.plot_drop_log(epochs.drop_log, show=False)
        drop_fig.savefig(sub_out / f"{sub_id}_drop_log.svg", dpi=600)
        plt.close(drop_fig)
        ar = autoreject.AutoReject(
            n_interpolate=np.array([1, 2, 4, 6, 8]),
            picks=["eeg"],
            thresh_method="random_search",
            random_state=42,
            n_jobs=1,
            verbose=False,
        )
        epochs_ar, reject_log = ar.fit_transform(epochs, return_log=True)
        reject_log_fig = reject_log.plot(orientation="horizontal", show=False)
        reject_log_fig.savefig(sub_out / f"{sub_id}_autoreject_log.svg", dpi=600)
        plt.close(reject_log_fig)
        epochs_ar.save(sub_out / f"{sub_id}_ar-epo.fif", overwrite=True)
        n_epochs_possible = len(epochs.drop_log)
        n_epochs_before_ar = len(epochs)
        n_epochs_after_ar = len(epochs_ar)
        stats["n_epochs_total"] = n_epochs_possible
        stats["n_epochs_before_autoreject"] = n_epochs_before_ar
        stats["n_epochs_after_autoreject"] = n_epochs_after_ar
        stats["n_epochs_dropped_by_annotation"] = n_epochs_possible - n_epochs_before_ar
        stats["n_epochs_autoreject_dropped"] = n_epochs_before_ar - n_epochs_after_ar
        stats["n_epochs_dropped"] = n_epochs_possible - n_epochs_after_ar
        stats["n_epochs_kept"] = n_epochs_after_ar
        stats["percent_epochs_kept"] = 100 * n_epochs_after_ar / n_epochs_possible
        return stats

    def process_one_subject(
        sub_id, group_key, group_info, datapath, derivpath, keep_iz
    ):
        try:
            stats = preprocess_subject(
                sub_id, group_key, group_info, datapath, derivpath, keep_iz
            )
            return (sub_id, stats, None)
        except Exception:
            error_text = traceback.format_exc()
            return (sub_id, None, error_text)

    for group_key, group_info in GROUPS.items():
        datapath = basepath / "RAW" / group_info["data_folder"]
        output = PREPROCESS_DIR / f"preprocess_{group_key}"
        output.mkdir(parents=True, exist_ok=True)
        subjects = sorted((p.name for p in datapath.glob("sub-*") if p.is_dir()))
        stats_file = output / f"process_stats_{group_key}.csv"
        previous = (
            pd.read_csv(stats_file, dtype={"subject": str})
            if stats_file.exists()
            else pd.DataFrame()
        )
        done = (
            set(previous.dropna(subset=["n_epochs_after_autoreject"])["subject"])
            if not previous.empty
            else set()
        )
        pending = [
            s
            for s in subjects
            if OVERWRITE
            or s not in done
            or (not (output / s / f"{s}_ar-epo.fif").exists())
        ]
        if not pending:
            continue
        keep_iz = all_subjects_have_iz(subjects, group_info, datapath)
        results = Parallel(n_jobs=N_JOBS)(
            (
                delayed(process_one_subject)(
                    s, group_key, group_info, datapath, output, keep_iz
                )
                for s in pending
            )
        )
        rows, errors = ([], [])
        for subject, stats, error in results:
            if error is None:
                rows.append({"subject": subject, **stats})
            else:
                errors.append({"subject": subject, "error": error})
        updated = pd.DataFrame(rows)
        if not previous.empty:
            updated = pd.concat(
                [previous.loc[~previous.subject.isin(pending)], updated],
                ignore_index=True,
            )
        updated.to_csv(stats_file, index=False)
        pd.DataFrame(errors, columns=["subject", "error"]).to_csv(
            output / "preprocessing_errors.csv", index=False
        )


if __name__ == "__main__":
    main()
