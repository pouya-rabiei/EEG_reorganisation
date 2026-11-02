"""Seven-network connectivity: boxplots, rainclouds, parcel heatmaps and connectomes."""

from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns
from nilearn import plotting
from config import (
    GROUP_LABELS, plot_results_dir, figure_dir, conn_labels, con_types,
    WITHIN_NETWORKS, WITHIN_NETWORK_LABELS, load_parcels, load_matrix,
)
from plotting_utils import save_plot, plot_network_boxes, plot_rainclouds

METRIC_LABELS = {"mean_within_connectivity": "Mean within-network connectivity"}


def plot_parcel_difference(difference, parcels, title, con, output_path):
    labels = [name.removeprefix("7Networks_") for name in parcels["ROI Name"]]
    coords = parcels[["R", "A", "S"]].to_numpy(float)
    limit = float(np.abs(difference).max()) or 1.
    norm = TwoSlopeNorm(0, -limit, limit)
    size = (20, 13.5) if len(parcels) > 18 else (17, 9)
    fig = plt.figure(figsize=size, facecolor="white")
    fig.suptitle(title, fontsize=22, y=.96)
    heat_ax = fig.add_axes([.13, .34, .32, .47])
    brain_ax = fig.add_axes([.51, .34, .46, .47])
    sns.heatmap(
        difference, mask=np.triu(np.ones_like(difference, dtype=bool)),
        cmap="turbo", norm=norm, square=True, linewidths=.7, linecolor="white",
        xticklabels=labels, yticklabels=labels, cbar=False, ax=heat_ax,
    )
    plt.setp(heat_ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor", fontsize=16)
    plt.setp(heat_ax.get_yticklabels(), rotation=0, fontsize=16)
    heat_ax.set_xlabel("Parcel", fontsize=20)
    heat_ax.set_ylabel("Parcel", fontsize=20)
    # Display the strongest 20% of absolute differences; this is not inference.
    plotting.plot_connectome(
        difference, coords, display_mode="lyrz", edge_threshold="80%",
        edge_cmap="turbo", edge_vmin=-limit, edge_vmax=limit,
        node_color="#F4A261", node_size=45,
        node_kwargs={"edgecolors": "#303030", "linewidths": .7},
        colorbar=False, annotate=False, black_bg=False, axes=brain_ax,
    )
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="turbo"),
        cax=fig.add_axes([.55, .22, .4, .025]), orientation="horizontal",
    )
    cbar.set_label(f"Mean {conn_labels[con]} difference (first − second)", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    save_plot(fig, output_path)


def main(distributions_only=False):
    plt.rcParams["svg.fonttype"] = "none"
    folder = plot_results_dir / "within_network_connectivity"
    data = pd.read_csv(folder / "subject_metrics.csv")
    data = data.dropna(
        subset=["sample_id", "age", "sex", "mean_within_connectivity"]
    )
    contrasts = pd.read_csv(folder / "GEE_contrasts.csv")
    estimates = pd.read_csv(folder / "GEE_adjusted_means.csv")
    output_dir = figure_dir / "within_network_connectivity" / "FIGURES"
    output_dir.mkdir(parents=True, exist_ok=True)
    for con in con_types:
        plot_network_boxes(data, con, "mean_within_connectivity",
                           "Mean within-network connectivity", output_dir, contrasts)
    plot_rainclouds(data, estimates, contrasts, output_dir, METRIC_LABELS)

    if distributions_only:
        return

    parcels = load_parcels()
    labels = parcels["ROI Name"].tolist()
    network_labels = parcels["ROI Name"].str.split("_").str[2]

    @lru_cache(maxsize=None)
    def group_mean(group, con, band):
        people = data.loc[
            data.analysis_group.eq(group) & data.con_type.eq(con) & data.freq_band.eq(band),
            ["group", "subject"],
        ].drop_duplicates()
        matrices = [load_matrix(source, subject, con, band, labels)
                    for source, subject in people.itertuples(index=False, name=None)]
        return np.mean(matrices, axis=0)

    keys = ["first_group", "second_group", "con_type", "band", "network"]
    locations = contrasts.loc[contrasts.network.isin(WITHIN_NETWORKS), keys].drop_duplicates()
    for first, second, con, band, network in locations.itertuples(index=False, name=None):
        idx = np.flatnonzero(network_labels.eq(network))
        difference = (group_mean(first, con, band) - group_mean(second, con, band))[np.ix_(idx, idx)]
        np.fill_diagonal(difference, 0)
        title = (f"{GROUP_LABELS[first]} vs {GROUP_LABELS[second]}\n"
                 f"{conn_labels[con]} · {band} · {WITHIN_NETWORK_LABELS[network]}")
        plot_parcel_difference(
            difference, parcels.iloc[idx], title, con,
            output_dir / f"{first}_vs_{second}_{network}_{band}_{con}_parcel_connectome",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--distributions-only", action="store_true")
    args = parser.parse_args()
    main(distributions_only=args.distributions_only)
