"""Within-network topology: distributions and significance-masked glass brains."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import nibabel as nib
from nilearn import datasets, plotting
from config import (
    GROUP_LABELS, plot_results_dir, figure_dir, conn_labels, con_types,
    TOPOLOGY_NETWORKS as NETWORK_ORDER, WITHIN_NETWORK_LABELS, load_parcels,
)
from plotting_utils import save_plot, plot_network_boxes, plot_rainclouds

METRIC_LABELS = {
    "within_network_clustering": "Clustering",
    "within_network_efficiency": "Efficiency",
}


def plot_brain_maps(contrasts, output_dir):
    supported = (
        contrasts.passes_interaction_gate.astype(str).str.lower().isin(["true", "1"])
        & contrasts.p_corrected.lt(.05)
        & contrasts.interaction_p_corrected.lt(.05)
    )
    rows_all = contrasts.loc[contrasts.network.ne("Limbic")].copy()
    limits = rows_all.groupby("metric").beta.apply(lambda values: values.abs().max())
    rows_all = rows_all.loc[supported.loc[rows_all.index] & np.isfinite(rows_all.beta)]
    if rows_all.empty:
        return
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=100, yeo_networks=7, resolution_mm=2, verbose=0
    )
    atlas_img = nib.load(atlas.maps)
    atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int16)
    lut = atlas.lut.loc[atlas.lut["index"].ne(0), ["index", "name"]].copy()
    if set(lut["name"]) != set(load_parcels()["ROI Name"]):
        raise ValueError("Schaefer atlas and study parcel names do not match")
    lut["network"] = lut["name"].str.split("_").str[2]
    masks = {
        network: np.isin(atlas_data, lut.loc[lut.network.eq(network), "index"])
        for network in NETWORK_ORDER if network in contrasts.network.unique()
    }
    keys = ["first_group", "second_group", "con_type", "band", "metric"]
    for (first, second, con, band, metric), rows in rows_all.groupby(keys, sort=False):
        limit = float(limits[metric]) or 1.
        volume = np.zeros(atlas_data.shape, dtype=np.float32)
        for row in rows.itertuples(index=False):
            if np.isfinite(row.beta):
                volume[masks[row.network]] = row.beta
        significant = rows.network.unique().tolist()
        names = ", ".join(WITHIN_NETWORK_LABELS[n] for n in significant)
        network_title = f"{names} {'network' if len(significant) == 1 else 'networks'}"
        fig, ax = plt.subplots(figsize=(12, 5.5))
        fig.suptitle(
            f"{GROUP_LABELS[first]} vs {GROUP_LABELS[second]}\n"
            f"{conn_labels[con]} · {band} · {METRIC_LABELS[metric]}",
            fontsize=20, y=.99,
        )
        fig.text(.5, .82, network_title, ha="center", fontsize=16)
        display = plotting.plot_glass_brain(
            nib.Nifti1Image(volume, atlas_img.affine), display_mode="lyrz",
            plot_abs=False, threshold=0, resampling_interpolation="nearest",
            cmap="turbo", vmin=-limit, vmax=limit, colorbar=False,
            annotate=False, black_bg=False, axes=ax,
        )
        # Outlines mark network-level significance, not parcel-level tests.
        for network in significant:
            display.add_contours(
                nib.Nifti1Image(masks[network].astype(np.float32), atlas_img.affine),
                threshold=None, levels=[.5], colors="black", linewidths=2.5,
            )
        cbar = fig.colorbar(
            ScalarMappable(norm=TwoSlopeNorm(0, -limit, limit), cmap="turbo"),
            cax=fig.add_axes([.2, .1, .6, .035]), orientation="horizontal",
        )
        cbar.set_label("Adjusted β (first − second)", fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        fig.subplots_adjust(left=.03, right=.97, top=.73, bottom=.23)
        filename = f"network_{first}_vs_{second}_{con}_{band}_{metric}_brain"
        save_plot(fig, output_dir / filename)


def main():
    plt.rcParams["svg.fonttype"] = "none"
    folder = plot_results_dir / "within_network_topology"
    data = pd.read_csv(folder / "subject_metrics.csv")
    contrasts = pd.read_csv(folder / "GEE_contrasts.csv")
    contrasts[["first_group", "second_group"]] = contrasts.comparison.str.split(" vs ", expand=True)
    estimates = pd.read_csv(folder / "GEE_adjusted_means.csv")
    data = data.loc[data.network.isin(NETWORK_ORDER)].copy()
    contrasts = contrasts.loc[contrasts.network.isin(NETWORK_ORDER)].copy()
    estimates = estimates.loc[estimates.network.isin(NETWORK_ORDER)].copy()
    output_dir = figure_dir / "within_network_topology" / "FIGURES"
    output_dir.mkdir(parents=True, exist_ok=True)
    for con in con_types:
        for metric, label in METRIC_LABELS.items():
            plot_network_boxes(data, con, metric, label, output_dir, contrasts)
    supported = contrasts.loc[
        contrasts.passes_interaction_gate.astype(str).str.lower().isin(["true", "1"])
        & contrasts.p_corrected.lt(.05)
        & contrasts.interaction_p_corrected.lt(.05)
    ]
    keys = ["con_type", "band", "metric", "network"]
    raincloud_estimates = estimates.merge(supported[keys].drop_duplicates(), on=keys)
    plot_rainclouds(data, raincloud_estimates, contrasts, output_dir, METRIC_LABELS)
    plot_brain_maps(contrasts, output_dir)


if __name__ == "__main__":
    main()
