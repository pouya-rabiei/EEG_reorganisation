"""Paths and settings shared by the numbered scripts."""

import os
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

basepath = Path(
    os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parent)
)
comparison_dir = Path(
    os.environ.get("EEG_RESULTS_DIR", basepath / "derivatives" / "GROUP_CONN_NETW")
)
spectral_dir = comparison_dir / "SPECTRAL"
plot_results_dir = Path(os.environ.get("EEG_PLOT_RESULTS_DIR", comparison_dir))
figure_dir = Path(os.environ.get("EEG_FIGURES_DIR", plot_results_dir))
plot_spectral_dir = plot_results_dir / "SPECTRAL"
PREPROCESS_DIR = Path(os.environ.get(
    "EEG_PREPROCESS_DIR", basepath / "derivatives" / "PREPROCESS_CONN"
))
SOURCE_DIR = Path(os.environ.get(
    "EEG_SOURCE_DIR", basepath / "derivatives" / "CONN_NETW"
))
centroid_file = Path(
    os.environ.get(
        "SCHAEFER_CENTROIDS",
        basepath.parent
        / "Codes"
        / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv",
    )
)
freq_bands = {
    "Theta": [4.0, 7.9],
    "Alpha": [8.0, 12.9],
    "Beta": [13.0, 29.9],
    "Gamma": [30.0, 45.0],
}
con_types = ["aec", "dwpli"]
NETWORK_ORDER = [
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
]
network_names = NETWORK_ORDER
WITHIN_NETWORKS = NETWORK_ORDER.copy()
TOPOLOGY_NETWORKS = [network for network in NETWORK_ORDER if network != "Limbic"]
WITHIN_NETWORK_LABELS = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience/Ventral Attention",
    "Default": "Default Mode",
    "Limbic": "Limbic",
    "Cont": "Control",
}
network_pairs = list(combinations(NETWORK_ORDER, 2))
ANALYSIS_GROUPS = ["HC", "NCBP_nontrans", "NCBP_trans"]
PLOT_GROUPS = ["NCBP_trans", "NCBP_nontrans", "HC"]
GROUP_LABELS = {
    "NCBP_trans": "Transitioners",
    "NCBP_nontrans": "Non-Transitioners",
    "HC": "Healthy Controls",
}
group_palette = {"NCBP_trans": "#D55E00", "NCBP_nontrans": "#009E73", "HC": "#0072B2"}
conn_labels = {"aec": "AEC", "dwpli": "dWPLI"}
comparison_pairs = [
    ("NCBP_trans", "NCBP_nontrans"),
    ("NCBP_trans", "HC"),
    ("NCBP_nontrans", "HC"),
]
comparison_specs = [
    {
        "comparison": f"{a}_vs_{b}",
        "comparison_display": f"{a} vs {b}",
        "first_group": a,
        "second_group": b,
    }
    for a, b in comparison_pairs
]
derivpath = {g: SOURCE_DIR / f"conn_{g}" for g in ["ncbp", "ncbphc"]}
N_JOBS = 4
OVERWRITE = False
MIN_EPOCHS = 50
RANGE = (1.0, 45.0)
N_RESAMPLES = 5000
SEED = 42
DENSITY = 0.20
NBS_THRESH_LIST = [3.0, 3.5, 4.0]
NBS_K = N_RESAMPLES
NBS_SEED = SEED
NBS_BATCH_SIZE = 128
NBS_ALPHA = 0.05
NBS_COMPARISONS = comparison_pairs
NBS_CON_TYPES = con_types
NBS_BANDS = list(freq_bands)
NBS_NETWORK_ORDER = NETWORK_ORDER
NBS_NETWORK_COLORS = {
    "Vis": "#1F77B4",
    "SomMot": "#FF7F0E",
    "DorsAttn": "#2CA02C",
    "SalVentAttn": "#D62728",
    "Limbic": "#9467BD",
    "Cont": "#8C564B",
    "Default": "#E377C2",
}


def load_subject_info():
    tables = []
    for group, folder in [("ncbp", "ncbp"), ("ncbphc", "ncbpmc")]:
        df = pd.read_csv(
            basepath / "RAW" / f"raw_data_{folder}" / f"demo_{folder}.csv",
            dtype={"subject": str},
        )
        df["subject"] = df["subject"].str.strip()
        df = df.dropna(subset=["subject"])
        df = df.loc[df.subject.ne("")].copy()
        df["group"] = group
        df["analysis_group"] = (
            "HC"
            if group == "ncbphc"
            else pd.to_numeric(df["chroncity_class_6m"]).map(
                {0: "NCBP_nontrans", 1: "NCBP_trans"}
            )
        )
        qc = pd.read_csv(
            PREPROCESS_DIR / f"preprocess_{group}" / f"process_stats_{group}.csv",
            dtype={"subject": str},
        )
        included = qc.loc[
            qc.n_epochs_after_autoreject.ge(MIN_EPOCHS), "subject"
        ].str.strip()
        tables.append(
            df.loc[
                df.subject.isin(included),
                ["subject", "group", "analysis_group", "age", "sex"],
            ]
        )
    info = pd.concat(tables, ignore_index=True).dropna(subset=["analysis_group"])
    if info.duplicated(["group", "subject"]).any():
        raise ValueError("Duplicate participant metadata")
    info["sample_id"] = info["group"] + "_" + info["subject"]
    return info


def load_parcels():
    return pd.read_csv(centroid_file)


def load_matrix(source_group, subject, con, band, labels):
    path = derivpath[source_group] / subject / f"{subject}_{con}_matrix_{band}.csv"
    matrix = pd.read_csv(path, index_col=0).loc[labels, labels].to_numpy(float)
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise ValueError(f"Expected a finite symmetric matrix: {path}")
    return matrix
