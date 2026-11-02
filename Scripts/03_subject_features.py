"""Subject level source spectra, connectivity, and graph metrics as well as group collection."""

from pathlib import Path
from os.path import join as opj
from glob import glob
import numpy as np
import pandas as pd
import bct
from joblib import Parallel, delayed
from config import (
    DENSITY,
    RANGE,
    NETWORK_ORDER,
    WITHIN_NETWORKS,
    TOPOLOGY_NETWORKS,
    N_JOBS,
    OVERWRITE,
    PREPROCESS_DIR,
    SOURCE_DIR,
    comparison_dir,
    con_types,
    derivpath,
    freq_bands,
    load_matrix,
    load_parcels,
    load_subject_info,
    network_names,
    network_pairs,
    spectral_dir,
)

def threshold_binary(matrix, density=DENSITY, within_network=False):
    """Retain the strongest proportion of unique edges, then binarize."""
    upper = np.triu_indices(len(matrix), 1)
    values = matrix[upper]
    count = int(np.floor(density * len(values)))
    # Preserve the established deterministic tie ordering for each analysis.
    order = (np.argsort(-values, kind="stable") if within_network
             else np.argsort(values, kind="stable")[::-1])
    selected = order[:count]
    if count == 0 or np.any(values[selected] <= 0):
        raise ValueError("Too few positive edges for the requested density")
    adjacency = np.zeros(matrix.shape, dtype=float)
    adjacency[upper[0][selected], upper[1][selected]] = 1
    adjacency += adjacency.T
    return adjacency, float(values[selected].min())


def graph_metrics(adjacency):
    # Keep genuine zeros; clustering is NA only if no node has two neighbours.
    clustering = (float(bct.clustering_coef_bu(adjacency).mean())
                  if np.any(adjacency.sum(axis=1) >= 2) else np.nan)
    efficiency = float(bct.efficiency_bin(adjacency, local=False))
    return clustering, efficiency


def collect_connectivity(subject_info=None):
    """Collect existing original matrices; no EEG/source processing is rerun."""
    info = load_subject_info() if subject_info is None else subject_info
    parcels = load_parcels()
    labels = parcels["ROI Name"].tolist()
    parcel_networks = parcels["ROI Name"].str.split("_").str[2]
    indices = {net: np.flatnonzero(parcel_networks.eq(net)) for net in NETWORK_ORDER}
    graph_rows, topology_rows, within_rows, between_rows, files = [], [], [], [], []
    for person in info.to_dict("records"):
        group, subject = person["group"], person["subject"]
        for con in con_types:
            for band in freq_bands:
                matrix = load_matrix(group, subject, con, band, labels)
                common = {**person, "con_type": con, "freq_band": band}
                files.append({
                    **common, "matrix_file": str(
                        derivpath[group] / subject / f"{subject}_{con}_matrix_{band}.csv"
                    ),
                })
                adjacency, cutoff = threshold_binary(matrix)
                clustering, efficiency = graph_metrics(adjacency)
                edges = int(adjacency.sum() / 2)
                graph_rows.append({
                    **common, "global_clustering": clustering,
                    "global_efficiency": efficiency, "threshold_value": cutoff,
                    "threshold_density_target": DENSITY,
                    "threshold_density_actual": edges / (len(matrix)*(len(matrix)-1)/2),
                    "n_edges": edges, "n_nodes": len(matrix),
                })
                for network in WITHIN_NETWORKS:
                    idx = indices[network]
                    block = matrix[np.ix_(idx, idx)]
                    upper = np.triu_indices(len(idx), 1)
                    # Threshold the original network submatrix, not the whole-brain graph.
                    network_adj, network_cutoff = threshold_binary(block, within_network=True)
                    clustering, efficiency = graph_metrics(network_adj)
                    within_rows.append({
                        **common, "network": network,
                        "mean_within_connectivity": float(block[upper].mean()),
                        "n_rois_in_network": len(idx),
                    })
                    topology_rows.append({
                        **common, "network": network,
                        "within_network_clustering": clustering,
                        "within_network_efficiency": efficiency,
                        "threshold_density_target": DENSITY,
                        "within_network_threshold_value": network_cutoff,
                        "subgraph_n_nodes": len(idx),
                        "subgraph_n_edges": int(network_adj.sum()/2),
                        "subgraph_density_actual": float(network_adj[upper].mean()),
                    })
                for first, second in network_pairs:
                    between_rows.append({
                        **common, "network_a": first, "network_b": second,
                        "network_pair": f"{first}__{second}",
                        "mean_between_connectivity": float(
                            matrix[np.ix_(indices[first], indices[second])].mean()
                        ),
                    })
    for folder_name, rows, filename in [
        ("whole_brain_topology", graph_rows, "whole_brain_metrics.csv"),
        ("within_network_topology", topology_rows, "subject_metrics.csv"),
        ("within_network_connectivity", within_rows, "subject_metrics.csv"),
        ("between_network_connectivity", between_rows, "subject_metrics.csv"),
    ]:
        folder = comparison_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(rows)
        if folder_name == "within_network_topology":
            table = table.loc[table.network.isin(TOPOLOGY_NETWORKS)]
        table.to_csv(folder / filename, index=False, na_rep="NA")
    info.to_csv(comparison_dir / "subject_info.csv", index=False)
    pd.DataFrame(files).to_csv(comparison_dir / "connectivity_files.csv", index=False)


def main(connectivity_only=False):
    import mne
    from mne.datasets import fetch_fsaverage
    from mne.beamformer import make_lcmv, apply_lcmv_epochs
    from mne.time_frequency import psd_array_multitaper
    from mne_connectivity import spectral_connectivity_epochs, envelope_correlation
    from fooof import FOOOF
    from fooof.utils import interpolate_spectrum

    def compute_source_psd_fooof_roi(
        vir_ts,
        sfreq,
        roi_names,
        fmin=RANGE[0],
        fmax=RANGE[1],
        bandwidth=2.0,
        adaptive=False,
        low_bias=True,
        peak_width_limits=(1.0, 12.0),
        max_n_peaks=6,
        min_peak_height=0.1,
        peak_threshold=2.0,
        aperiodic_mode="fixed",
        line_noise_interp_ranges=None,
        line_noise_buffer=3,
    ):
        fooof_rows = []
        peak_rows = []
        flattened_rows = []
        roi_psd_raw_dict = {}
        roi_psd_interp_dict = {}
        for roi_idx, roi_name in enumerate(roi_names):
            roi_data = vir_ts[:, roi_idx, :]
            psds_full, freqs_full = psd_array_multitaper(
                roi_data,
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                bandwidth=bandwidth,
                adaptive=adaptive,
                low_bias=low_bias,
                normalization="length",
                remove_dc=True,
                output="power",
                n_jobs=1,
                verbose=False,
            )
            psd_mean_raw = psds_full.mean(axis=0)
            roi_psd_raw_dict[roi_name] = psd_mean_raw
            freqs_interp = freqs_full.copy()
            psd_mean_interp = psd_mean_raw.copy()
            if line_noise_interp_ranges is not None:
                for interp_range in line_noise_interp_ranges:
                    lo, hi = interp_range
                    if lo >= freqs_interp.min() and hi <= freqs_interp.max():
                        interp_out = interpolate_spectrum(
                            freqs_interp,
                            psd_mean_interp,
                            interp_range=[lo, hi],
                            buffer=line_noise_buffer,
                        )
                        if isinstance(interp_out, tuple):
                            freqs_interp, psd_mean_interp = interp_out
                        else:
                            psd_mean_interp = interp_out
            roi_psd_interp_dict[roi_name] = psd_mean_interp
            fooof_mask = (freqs_interp >= fmin) & (freqs_interp <= fmax)
            freqs_fooof = freqs_interp[fooof_mask]
            psd_fooof = psd_mean_interp[fooof_mask]
            valid = np.isfinite(freqs_fooof) & np.isfinite(psd_fooof) & (psd_fooof > 0)
            freqs_fooof = freqs_fooof[valid]
            psd_fooof = psd_fooof[valid]
            if len(freqs_fooof) < 3:
                continue
            fm = FOOOF(
                peak_width_limits=peak_width_limits,
                max_n_peaks=max_n_peaks,
                min_peak_height=min_peak_height,
                peak_threshold=peak_threshold,
                aperiodic_mode=aperiodic_mode,
                verbose=False,
            )
            try:
                fm.fit(freqs_fooof, psd_fooof)
            except Exception as e:
                continue
            if not fm.has_model:
                continue
            flattened_log = fm.power_spectrum - fm._ap_fit
            for f, flat_val in zip(fm.freqs, flattened_log):
                flattened_rows.append(
                    {
                        "roi": roi_name,
                        "frequency": float(f),
                        "flattened_log_power": float(flat_val),
                    }
                )
            row = {
                "roi": roi_name,
                "offset": float(fm.aperiodic_params_[0]),
                "exponent": float(fm.aperiodic_params_[1]),
                "r2": float(fm.r_squared_),
                "error": float(fm.error_),
                "n_peaks": int(len(fm.peak_params_)),
                "flattened_mean": float(np.mean(flattened_log)),
                "flattened_min": float(np.min(flattened_log)),
                "flattened_max": float(np.max(flattened_log)),
            }
            for band_name, (b_lo, b_hi) in freq_bands.items():
                band_key = str(band_name).lower()
                band_peaks = [p for p in fm.peak_params_ if b_lo <= p[0] <= b_hi]
                if band_peaks:
                    peak = max(band_peaks, key=lambda x: x[1])
                    row[f"{band_key}_cf"] = float(peak[0])
                    row[f"{band_key}_pw"] = float(peak[1])
                    row[f"{band_key}_bw"] = float(peak[2])
                else:
                    row[f"{band_key}_cf"] = np.nan
                    row[f"{band_key}_pw"] = np.nan
                    row[f"{band_key}_bw"] = np.nan
            fooof_rows.append(row)
            for peak_idx, peak in enumerate(fm.peak_params_):
                cf, pw, bw = peak
                band_label = "other"
                if 4.0 <= cf <= 7.9:
                    band_label = "theta"
                elif 8.0 <= cf <= 12.9:
                    band_label = "alpha"
                elif 13.0 <= cf <= 29.9:
                    band_label = "beta"
                elif 30.0 <= cf <= 45.0:
                    band_label = "gamma"
                peak_rows.append(
                    {
                        "roi": roi_name,
                        "peak_index": int(peak_idx),
                        "cf": float(cf),
                        "pw": float(pw),
                        "bw": float(bw),
                        "band": band_label,
                    }
                )
        fooof_df = pd.DataFrame(fooof_rows)
        peaks_df = pd.DataFrame(
            peak_rows, columns=["roi", "peak_index", "cf", "pw", "bw", "band"]
        )
        flattened_df = pd.DataFrame(
            flattened_rows, columns=["roi", "frequency", "flattened_log_power"]
        )
        roi_psd_raw_df = pd.DataFrame(roi_psd_raw_dict, index=freqs_full)
        roi_psd_raw_df.index.name = "frequency"
        roi_psd_interp_df = pd.DataFrame(roi_psd_interp_dict, index=freqs_interp)
        roi_psd_interp_df.index.name = "frequency"
        return (fooof_df, peaks_df, roi_psd_raw_df, roi_psd_interp_df, flattened_df)

    def band_specific_source_stcs(epochs, forward, l_freq, h_freq):
        ep_band = epochs.copy().filter(
            l_freq=l_freq, h_freq=h_freq, n_jobs=1, verbose=False
        )
        data_cov = mne.compute_covariance(
            ep_band, method="empirical", keep_sample_mean=True, verbose=False
        )
        rank = mne.compute_rank(ep_band, rank="info", verbose=False)
        filters = make_lcmv(
            ep_band.info,
            forward,
            data_cov,
            reg=0.05,
            pick_ori="max-power",
            weight_norm="unit-noise-gain",
            rank=rank,
            verbose=False,
        )
        stcs = apply_lcmv_epochs(
            ep_band, filters, return_generator=False, verbose=False
        )
        del ep_band, data_cov, filters
        return stcs

    def band_connectivity(stcs, sfreq, l_freq, h_freq):
        con = spectral_connectivity_epochs(
            data=stcs,
            method="wpli2_debiased",
            mode="multitaper",
            sfreq=sfreq,
            fmin=l_freq,
            fmax=h_freq,
            faverage=True,
            n_jobs=1,
            verbose=False,
        )
        dwpli = con.get_data(output="dense")[:, :, 0]
        dwpli_lower = np.tril(dwpli, k=-1)
        dwpli = dwpli_lower + dwpli_lower.T
        np.fill_diagonal(dwpli, 0.0)
        vtcs = np.stack([stc.data for stc in stcs], axis=0)
        aec = (
            envelope_correlation(
                data=vtcs,
                orthogonalize="pairwise",
                log=False,
                absolute=True,
                verbose=False,
            )
            .combine()
            .get_data(output="dense")[:, :, 0]
        )
        aec /= 0.577
        np.fill_diagonal(aec, 0.0)
        return (dwpli.astype(np.float32), aec.astype(np.float32))

    def source_connectivity_all_bands(
        epochs, forward, src, labels, sub_out, sub_id, freq_bands=freq_bands
    ):
        sfreq = epochs.info["sfreq"]
        rois = np.asarray(labels, dtype=str)
        n_rois = len(rois)
        n_sources = sum((source_space["nuse"] for source_space in src))
        if n_sources != n_rois:
            raise ValueError(
                f"Source space contains {n_sources} sources, but {n_rois} ROI labels were provided"
            )
        if forward["nsource"] != n_rois:
            raise ValueError(
                f"Forward solution contains {forward['nsource']} sources, but {n_rois} ROI labels were provided"
            )
        for band, (l_freq, h_freq) in freq_bands.items():
            stcs = band_specific_source_stcs(
                epochs=epochs, forward=forward, l_freq=l_freq, h_freq=h_freq
            )
            dwpli, aec = band_connectivity(
                stcs=stcs, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq
            )
            matrices = {"dwpli": dwpli, "aec": aec}
            for name, matrix in matrices.items():
                if matrix.shape != (n_rois, n_rois):
                    raise ValueError(
                        f"{name} matrix has shape {matrix.shape}; expected {(n_rois, n_rois)}"
                    )
                if not np.isfinite(matrix).all():
                    raise ValueError(
                        f"{name} matrix for {band} contains non-finite values"
                    )
                if not np.allclose(matrix, matrix.T, rtol=1e-05, atol=1e-07):
                    raise ValueError(f"{name} matrix for {band} is not symmetric")
                output_file = sub_out / f"{sub_id}_{name}_matrix_{band}.csv"
                pd.DataFrame(matrix, index=rois, columns=rois).to_csv(output_file)
            del stcs, dwpli, aec

    def expected_subject_outputs(sub_out, sub_id):
        expected = [
            sub_out / f"{sub_id}_source_fooof_roi.csv",
            sub_out / f"{sub_id}_source_fooof_peaks.csv",
            sub_out / f"{sub_id}_source_psd_roi_raw.csv",
            sub_out / f"{sub_id}_source_psd_roi_interp.csv",
            sub_out / f"{sub_id}_source_fooof_network_summary.csv",
            sub_out / f"{sub_id}_source_fooof_flattened.csv",
            sub_out / f"{sub_id}_source_fooof_flattened_network.csv",
        ]
        if connectivity_only:
            expected = []
        for band in freq_bands:
            expected.append(sub_out / f"{sub_id}_dwpli_matrix_{band}.csv")
            expected.append(sub_out / f"{sub_id}_aec_matrix_{band}.csv")
        return expected

    def subject_already_done(sub_out, sub_id):
        return all(
            (path.exists() for path in expected_subject_outputs(sub_out, sub_id))
        )

    def process_one_subject_source_conn(
        sub_id, group, datapath, derivpath, line_noise_interp_ranges, overwrite=False
    ):
        sub_out = derivpath / sub_id
        sub_out.mkdir(parents=True, exist_ok=True)
        if subject_already_done(sub_out, sub_id) and (not overwrite):
            return {"group": group, "subject": sub_id, "status": "skipped", "error": ""}
        try:
            epoch_file = Path(datapath) / sub_id / f"{sub_id}_5s-epo.fif"
            if not epoch_file.exists():
                epoch_files = sorted(glob(opj(datapath, sub_id, "*_ar-epo.fif")))
                if len(epoch_files) != 1:
                    raise FileNotFoundError(
                        f"Expected {epoch_file.name} or one legacy *_ar-epo.fif"
                    )
                epoch_file = epoch_files[0]
            epoch = mne.read_epochs(epoch_file, preload=True, verbose=False)
            epoch.pick("eeg", exclude="bads")
            epoch.set_eeg_reference("average", projection=True, verbose=False)
            forward = mne.make_forward_solution(
                epoch.info,
                trans="fsaverage",
                src=src,
                bem=bem,
                eeg=True,
                meg=False,
                verbose=False,
            )
            if forward["nsource"] != sum((s["nuse"] for s in src)):
                raise RuntimeError("Forward source count does not match source space.")
            if connectivity_only:
                source_connectivity_all_bands(
                    epoch, forward, src, labels, sub_out, sub_id
                )
                return {"group": group, "subject": sub_id, "status": "success", "error": ""}
            data_cov = mne.compute_covariance(
                epoch, method="empirical", keep_sample_mean=True, verbose=False
            )
            rank = mne.compute_rank(epoch, rank="info", verbose=False)
            filters = make_lcmv(
                epoch.info,
                forward,
                data_cov,
                reg=0.05,
                pick_ori="max-power",
                weight_norm="unit-noise-gain",
                rank=rank,
                verbose=False,
            )
            stcs = apply_lcmv_epochs(
                epoch, filters, return_generator=True, verbose=False
            )
            roi = np.asarray(labels, dtype=str)
            vir_ts = np.stack([stc.data for stc in stcs], axis=0)
            if vir_ts.shape[1] != len(roi):
                raise ValueError(
                    f"STC contains {vir_ts.shape[1]} sources, but {len(roi)} labels were provided"
                )
            fooof_df, peaks_df, roi_psd_raw_df, roi_psd_interp_df, flattened_df = (
                compute_source_psd_fooof_roi(
                    vir_ts=vir_ts,
                    sfreq=epoch.info["sfreq"],
                    roi_names=roi,
                    fmin=RANGE[0],
                    fmax=RANGE[1],
                    bandwidth=2.0,
                    adaptive=False,
                    low_bias=True,
                    peak_width_limits=(0.5, 12.0),
                    max_n_peaks=6,
                    min_peak_height=0.1,
                    peak_threshold=2.0,
                    aperiodic_mode="fixed",
                    line_noise_interp_ranges=line_noise_interp_ranges,
                    line_noise_buffer=3,
                )
            )
            fooof_df.to_csv(sub_out / f"{sub_id}_source_fooof_roi.csv", index=False)
            peaks_df.to_csv(sub_out / f"{sub_id}_source_fooof_peaks.csv", index=False)
            roi_psd_raw_df.to_csv(sub_out / f"{sub_id}_source_psd_roi_raw.csv")
            roi_psd_interp_df.to_csv(sub_out / f"{sub_id}_source_psd_roi_interp.csv")
            flattened_df.to_csv(
                sub_out / f"{sub_id}_source_fooof_flattened.csv", index=False
            )
            net_rows = []
            for net in network_names:
                mask = fooof_df["roi"].str.contains(f"_{net}_", regex=False)
                if not mask.any():
                    continue
                tmp = fooof_df.loc[mask].copy()
                net_rows.append(
                    {
                        "network": net,
                        "n_rois": int(mask.sum()),
                        "offset_mean": tmp["offset"].mean(),
                        "offset_sd": tmp["offset"].std(),
                        "exponent_mean": tmp["exponent"].mean(),
                        "exponent_sd": tmp["exponent"].std(),
                        "alpha_cf_mean": tmp["alpha_cf"].mean(),
                        "alpha_cf_sd": tmp["alpha_cf"].std(),
                        "alpha_pw_mean": tmp["alpha_pw"].mean(),
                        "alpha_pw_sd": tmp["alpha_pw"].std(),
                    }
                )
            pd.DataFrame(net_rows).to_csv(
                sub_out / f"{sub_id}_source_fooof_network_summary.csv", index=False
            )
            flat_net_rows = []
            for net in network_names:
                mask = flattened_df["roi"].str.contains(f"_{net}_", regex=False)
                if not mask.any():
                    continue
                tmp = flattened_df.loc[mask].copy()
                tmp_net = tmp.groupby("frequency", as_index=False)[
                    "flattened_log_power"
                ].mean()
                tmp_net.insert(0, "network", net)
                flat_net_rows.append(tmp_net)
            if flat_net_rows:
                flattened_network_df = pd.concat(flat_net_rows, ignore_index=True)
            else:
                flattened_network_df = pd.DataFrame(
                    columns=["network", "frequency", "flattened_log_power"]
                )
            flattened_network_df.to_csv(
                sub_out / f"{sub_id}_source_fooof_flattened_network.csv", index=False
            )
            del stcs, vir_ts, filters, data_cov
            source_connectivity_all_bands(
                epochs=epoch,
                forward=forward,
                src=src,
                labels=labels,
                sub_out=sub_out,
                sub_id=sub_id,
                freq_bands=freq_bands,
            )
            return {"group": group, "subject": sub_id, "status": "success", "error": ""}
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            return {
                "group": group,
                "subject": sub_id,
                "status": "failed",
                "error": error_message,
            }

    subject_info = load_subject_info()
    metadata_dir = SOURCE_DIR if connectivity_only else spectral_dir
    metadata_dir.mkdir(parents=True, exist_ok=True)
    subject_info.to_csv(metadata_dir / "subject_info.csv", index=False)
    pos = load_parcels()
    labels = pos["ROI Name"].astype(str).to_numpy()
    network_to_indices = {
        net: np.flatnonzero(pos["ROI Name"].str.split("_").str[2].eq(net))
        for net in NETWORK_ORDER
    }
    rr = pos[["R", "A", "S"]].to_numpy(float) / 1000
    if OVERWRITE or any(
        (
            not subject_already_done(derivpath[row.group] / row.subject, row.subject)
            for row in subject_info.itertuples()
        )
    ):
        fs_dir = Path(fetch_fsaverage(verbose=False))
        src = mne.setup_volume_source_space(
            "fsaverage",
            pos={"rr": rr, "nn": rr / np.linalg.norm(rr, axis=1, keepdims=True)},
            subjects_dir=fs_dir.parent,
            verbose=False,
        )
        bem = str(fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif")
    status = []
    status_file = (SOURCE_DIR if connectivity_only else comparison_dir) / "subject_processing_status.csv"
    for group, participants in subject_info.groupby("group", sort=False):
        derivpath[group].mkdir(parents=True, exist_ok=True)
        for result in Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(
                (
                    delayed(process_one_subject_source_conn)(
                        s,
                        group,
                        PREPROCESS_DIR / f"preprocess_{group}",
                        derivpath[group],
                        ((58.0, 62.0),),
                        overwrite=OVERWRITE,
                    )
                    for s in participants.subject
                )
            ):
            status.append(result)
            pd.DataFrame(status).to_csv(status_file, index=False)
            print(f"{len(status)}/{len(subject_info)} {group} {result['subject']}: {result['status']} {result['error']}", flush=True)
    if any((row["status"] == "failed" for row in status)):
        raise RuntimeError(
            "Subject extraction failed; see subject_processing_status.csv"
        )
    if connectivity_only:
        return
    spectral_tables = {
        name: []
        for name in ["fooof_network_summary", "fooof_roi", "fooof_peak", "psd_interp"]
    }
    for person in subject_info.to_dict("records"):
        group, sub = (person["group"], person["subject"])
        folder = derivpath[group] / sub
        for name, suffix in [
            ("fooof_network_summary", "fooof_network_summary"),
            ("fooof_roi", "fooof_roi"),
            ("fooof_peak", "fooof_peaks"),
        ]:
            table = pd.read_csv(folder / f"{sub}_source_{suffix}.csv")
            spectral_tables[name].append(table.assign(**person))
        psd = pd.read_csv(folder / f"{sub}_source_psd_roi_interp.csv")
        psd["frequency"] = psd["frequency"].round(6)
        spectral_tables["psd_interp"].append(
            psd.melt(id_vars="frequency", var_name="roi", value_name="psd").assign(
                **person
            )
        )
    for name, tables in spectral_tables.items():
        pd.concat(tables, ignore_index=True).to_csv(
            spectral_dir / f"{name}_all_groups.csv", index=False
        )
    collect_connectivity(subject_info)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-connectivity", action="store_true")
    parser.add_argument("--connectivity-only", action="store_true")
    args = parser.parse_args()
    if args.collect_connectivity:
        collect_connectivity()
    else:
        main(connectivity_only=args.connectivity_only)
