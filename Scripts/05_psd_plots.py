"""PSD heatmaps."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import ANALYSIS_GROUPS, GROUP_LABELS, NETWORK_ORDER, spectral_dir


def main():
    psd_all = pd.read_csv(spectral_dir / "psd_interp_all_groups.csv")
    psd_cluster_summary = pd.read_csv(
        spectral_dir / "PSD_network_cluster_summary.csv"
    )
    psd_cluster_all = pd.read_csv(
        spectral_dir / "PSD_network_cluster_all.csv"
    )
    psd_fig_dir = spectral_dir / "PSD_HEATMAPS"
    psd_fig_dir.mkdir(parents=True, exist_ok=True)
    ANALYSIS_GROUPS = ["NCBP_trans", "NCBP_nontrans", "HC"]
    panel_pairs = [
        ("NCBP_trans", "NCBP_nontrans"),
        ("NCBP_trans", "HC"),
        ("NCBP_nontrans", "HC"),
    ]
    FREQ_MIN = 1.0
    FREQ_MAX = 45.0
    POWER_CMAP = "turbo"
    DIFFERENCE_CMAP = "turbo"
    psd_plot = psd_all[
        ["analysis_group", "sample_id", "roi", "frequency", "psd"]
    ].copy()
    psd_plot["frequency"] = pd.to_numeric(psd_plot["frequency"], errors="raise").round(
        6
    )
    psd_plot["psd"] = pd.to_numeric(psd_plot["psd"], errors="raise")
    psd_plot = psd_plot.loc[
        psd_plot["analysis_group"].isin(ANALYSIS_GROUPS)
        & psd_plot["frequency"].between(FREQ_MIN, FREQ_MAX)
    ].copy()
    if psd_plot.empty:
        raise ValueError("No PSD data are available for the selected groups.")
    if not np.isfinite(psd_plot["psd"]).all() or psd_plot["psd"].le(0).any():
        raise ValueError("PSD values must be finite and positive.")
    psd_plot["log10_psd"] = np.log10(psd_plot["psd"])

    def roi_network(roi_name):
        return str(roi_name).split("_")[2]

    first_seen = {roi: index for index, roi in enumerate(pd.unique(psd_plot["roi"]))}
    roi_order = sorted(
        first_seen,
        key=lambda roi: (
            NETWORK_ORDER.index(roi_network(roi)),
            0 if "_LH_" in str(roi) else 1,
            first_seen[roi],
        ),
    )
    network_indices = {
        network: [
            index for index, roi in enumerate(roi_order) if roi_network(roi) == network
        ]
        for network in NETWORK_ORDER
    }
    network_midpoints = [np.mean(network_indices[network]) for network in NETWORK_ORDER]
    network_boundaries = [
        max(network_indices[network]) + 0.5 for network in NETWORK_ORDER[:-1]
    ]
    subject_roi_psd = psd_plot.groupby(
        ["analysis_group", "sample_id", "roi", "frequency"],
        observed=True,
        as_index=False,
    )["log10_psd"].mean()
    group_roi_psd = subject_roi_psd.groupby(
        ["analysis_group", "roi", "frequency"], observed=True, as_index=False
    )["log10_psd"].mean()
    group_psd_matrices = {}
    psd_frequencies = None
    for group in ANALYSIS_GROUPS:
        rows = group_roi_psd.loc[group_roi_psd["analysis_group"].eq(group)]
        if rows.empty:
            raise ValueError(f"No PSD data found for {group}.")
        matrix_df = (
            rows.pivot(index="roi", columns="frequency", values="log10_psd")
            .reindex(index=roi_order)
            .sort_index(axis=1)
        )
        if matrix_df.isna().any().any():
            raise ValueError(f"Incomplete ROI-frequency grid for {group}.")
        current_frequencies = matrix_df.columns.to_numpy(dtype=float)
        if psd_frequencies is None:
            psd_frequencies = current_frequencies
        elif not np.array_equal(psd_frequencies, current_frequencies):
            raise ValueError("Frequency bins differ between groups.")
        group_psd_matrices[group] = matrix_df.to_numpy(dtype=float)
    difference_matrices = {
        (first, second): group_psd_matrices[first] - group_psd_matrices[second]
        for first, second in panel_pairs
    }
    all_group_values = np.concatenate(
        [matrix.ravel() for matrix in group_psd_matrices.values()]
    )
    power_vmin, power_vmax = np.percentile(all_group_values, [1, 99])
    all_difference_values = np.concatenate(
        [matrix.ravel() for matrix in difference_matrices.values()]
    )
    difference_limit = np.percentile(np.abs(all_difference_values), 99)
    if difference_limit == 0:
        difference_limit = 1.0
    sig_psd_networks = psd_cluster_summary.loc[
        pd.to_numeric(psd_cluster_summary["p_corrected"], errors="coerce").lt(0.05),
        ["comparison", "network"],
    ].drop_duplicates()
    sig_psd_clusters = psd_cluster_all.merge(
        sig_psd_networks,
        on=["comparison", "network"],
        how="inner",
        validate="many_to_one",
    )
    sig_psd_clusters = sig_psd_clusters.loc[
        pd.to_numeric(sig_psd_clusters["p_cluster"], errors="coerce").lt(0.05)
    ].copy()
    frequency_step = float(np.median(np.diff(psd_frequencies)))
    extent = [
        psd_frequencies.min() - frequency_step / 2,
        psd_frequencies.max() + frequency_step / 2,
        len(roi_order) - 0.5,
        -0.5,
    ]

    def add_psd_cluster_boxes(ax, comparison):
        rows = sig_psd_clusters.loc[sig_psd_clusters["comparison"].eq(comparison)]
        for cluster in rows.itertuples(index=False):
            indices = network_indices[cluster.network]
            x_min = max(float(cluster.freq_min) - frequency_step / 2, extent[0])
            x_max = min(float(cluster.freq_max) + frequency_step / 2, extent[1])
            if x_max <= x_min:
                continue
            y_min = min(indices) - 0.5
            y_max = max(indices) + 0.5
            ax.add_patch(
                Rectangle(
                    (x_min, y_min),
                    width=x_max - x_min,
                    height=y_max - y_min,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.5,
                    zorder=10,
                    clip_on=False,
                )
            )

    with plt.rc_context(
        {
            "font.size": 18,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    ):
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 18))
        fig.subplots_adjust(
            left=0.13, right=0.93, bottom=0.06, top=0.94, wspace=0.75, hspace=0.55
        )
        for row, (first, second) in enumerate(panel_pairs):
            comparison = f"{first} vs {second}"
            matrices = [
                group_psd_matrices[first],
                group_psd_matrices[second],
                difference_matrices[first, second],
            ]
            titles = [
                GROUP_LABELS[first],
                GROUP_LABELS[second],
                f"{GROUP_LABELS[first]}\nvs {GROUP_LABELS[second]}",
            ]
            for col, matrix in enumerate(matrices):
                ax = axes[row, col]
                ax.set_title(titles[col], fontsize=18, fontweight="bold", pad=12)
                is_difference = col == 2
                image = ax.imshow(
                    matrix,
                    origin="upper",
                    extent=extent,
                    aspect="auto",
                    interpolation="nearest",
                    cmap=DIFFERENCE_CMAP if is_difference else POWER_CMAP,
                    vmin=-difference_limit if is_difference else power_vmin,
                    vmax=difference_limit if is_difference else power_vmax,
                )
                for boundary in network_boundaries:
                    ax.axhline(boundary, color="white", linewidth=1.5, alpha=0.85)
                ax.set_yticks(network_midpoints)
                if col == 0:
                    ax.set_yticklabels(NETWORK_ORDER, fontsize=18)
                    ax.set_ylabel("Network", fontsize=20, labelpad=12)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0, labelsize=18)
                ax.tick_params(axis="x", labelsize=18)
                ax.set_xticks([10, 20, 30, 40])
                ax.set_xlabel("Frequency (Hz)", fontsize=20, labelpad=8)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if is_difference:
                    add_psd_cluster_boxes(ax, comparison)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.09)
                cbar = fig.colorbar(image, cax=cax)
                cbar.ax.tick_params(labelsize=18)
                cbar.outline.set_linewidth(0.5)
                cbar.set_label(
                    (
                        "Difference in mean log₁₀(PSD)"
                        if is_difference
                        else "Mean log₁₀(PSD)"
                    ),
                    fontsize=20,
                    labelpad=12,
                )
            axes[row, 0].text(
                -0.38,
                1.13,
                "ABC"[row],
                transform=axes[row, 0].transAxes,
                fontsize=40,
                fontweight="bold",
                va="bottom",
                clip_on=False,
            )
        fig.savefig(
            psd_fig_dir / "PSD_heatmap_panels.svg",
            dpi=600,
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
