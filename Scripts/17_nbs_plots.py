"""NBS heatmaps and connectomes from saved results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from nilearn import plotting
from config import (
    GROUP_LABELS,
    NBS_NETWORK_COLORS,
    NBS_NETWORK_ORDER,
    comparison_dir,
    conn_labels,
    load_parcels,
)


def main():

    def plot_nbs_result(
        t_matrix,
        significant_mask,
        parcel_info,
        first,
        second,
        con,
        band,
        threshold,
        heatmap_file,
        connectome_file,
    ):
        networks = parcel_info["ROI Name"].astype(str).str.split("_").str[2].to_numpy()
        order = np.concatenate(
            [np.flatnonzero(networks == net) for net in NBS_NETWORK_ORDER]
        )
        if len(order) != len(networks):
            raise ValueError("An NBS parcel has an unrecognized Schaefer network")
        sizes = np.array([(networks == net).sum() for net in NBS_NETWORK_ORDER])
        boundaries = sizes.cumsum()
        centres = boundaries - sizes / 2
        ordered_t = t_matrix[np.ix_(order, order)]
        ordered_sig = significant_mask[np.ix_(order, order)]
        triangle = np.triu(np.ones_like(ordered_sig, dtype=bool))
        limit = float(np.max(np.abs(t_matrix))) or 1.0
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        cmap = plt.get_cmap("turbo")
        n = len(order)
        has_effect = bool(significant_mask.any())
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.subplots_adjust(left=0.19, bottom=0.22, right=0.83, top=0.83)
        fig.suptitle(
            f"{GROUP_LABELS[first]} vs {GROUP_LABELS[second]}",
            fontsize=18,
            fontweight="bold",
            y=0.97,
        )
        ax.set_title(
            f"{conn_labels[con]} · {band} · NBS |t| > {threshold:g}",
            fontsize=18,
            pad=14,
        )
        ax.imshow(
            np.ma.array(ordered_t, mask=triangle),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            extent=(0, n, n, 0),
            alpha=0.25,
        )
        if has_effect:
            ax.imshow(
                np.ma.array(ordered_t, mask=triangle | ~ordered_sig),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                extent=(0, n, n, 0),
            )
        for boundary in boundaries[:-1]:
            ax.plot([0, boundary], [boundary, boundary], color="0.3", linewidth=0.7)
            ax.plot([boundary, boundary], [boundary, n], color="0.3", linewidth=0.7)
        ax.set_xticks(centres, NBS_NETWORK_ORDER, rotation=30, ha="right", fontsize=16)
        ax.set_yticks(centres, NBS_NETWORK_ORDER, fontsize=16)
        ax.set_xlabel("Network", fontsize=18)
        ax.set_ylabel("Network", fontsize=18)
        scalar = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(scalar, cax=fig.add_axes([0.86, 0.25, 0.025, 0.54]))
        cbar.set_label("Edgewise t", fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        fig.savefig(heatmap_file, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(13, 6.5))
        fig.subplots_adjust(left=0.03, right=0.9, top=0.78, bottom=0.26)
        fig.suptitle(
            f"{GROUP_LABELS[first]} vs {GROUP_LABELS[second]}",
            fontsize=22,
            fontweight="bold",
            y=0.97,
        )
        fig.text(
            0.5,
            0.87,
            f"{conn_labels[con]} · {band} · NBS |t| > {threshold:g}",
            ha="center",
            fontsize=20,
        )
        plotting.plot_connectome(
            np.where(significant_mask, t_matrix, 0.0),
            parcel_info[["R", "A", "S"]].to_numpy(dtype=float),
            display_mode="lyrz",
            node_size=30,
            node_color=[NBS_NETWORK_COLORS[net] for net in networks],
            edge_cmap=cmap,
            edge_vmin=-limit,
            edge_vmax=limit,
            edge_threshold=None,
            colorbar=False,
            annotate=False,
            black_bg=False,
            axes=ax,
        )
        cbar = fig.colorbar(scalar, cax=fig.add_axes([0.93, 0.29, 0.015, 0.4]))
        cbar.set_label("Edgewise t", fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        fig.legend(
            handles=[
                Patch(color=NBS_NETWORK_COLORS[net], label=net)
                for net in NBS_NETWORK_ORDER
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.09),
            ncol=7,
            frameon=False,
            fontsize=13,
        )
        fig.savefig(connectome_file, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    parcel_info = load_parcels()
    labels = parcel_info["ROI Name"].tolist()
    edges = pd.read_csv(comparison_dir / "nbs_edges_all_pairs_thresholds.csv")
    components = pd.read_csv(comparison_dir / "nbs_components_all_pairs_thresholds.csv")
    keys = [
        "first_group",
        "second_group",
        "con_type",
        "freq_band",
        "primary_threshold_t",
    ]
    for first, second, con, band, threshold in (
        components[keys].drop_duplicates().itertuples(index=False, name=None)
    ):
        comparison = f"{first} vs {second}"
        stem = f"{comparison}_{con}_{band}_t{threshold:.1f}"
        test_dir = comparison_dir / "nbs" / comparison / f"nbs_t_{threshold:.1f}"
        figure_dir = (
            comparison_dir / "nbs" / comparison / f"figures_nbs_t_{threshold:.1f}"
        )
        figure_dir.mkdir(parents=True, exist_ok=True)
        matrix = (
            pd.read_csv(test_dir / f"edgewise_t_matrix_{stem}.csv", index_col=0)
            .loc[labels, labels]
            .to_numpy(float)
        )
        selected = edges.loc[
            edges.comparison.eq(comparison)
            & edges.con_type.eq(con)
            & edges.freq_band.eq(band)
            & edges.primary_threshold_t.eq(threshold)
            & edges.significant.eq(True)
        ]
        significant = np.zeros_like(matrix, dtype=bool)
        index = {name: i for i, name in enumerate(labels)}
        for edge in selected.itertuples():
            i, j = (index[edge.roi_i], index[edge.roi_j])
            significant[i, j] = significant[j, i] = True
        plot_nbs_result(
            matrix,
            significant,
            parcel_info,
            first,
            second,
            con,
            band,
            threshold,
            figure_dir / f"heatmap_nbs_{stem}.svg",
            figure_dir / f"connectome_nbs_{stem}.svg",
        )


if __name__ == "__main__":
    main()
