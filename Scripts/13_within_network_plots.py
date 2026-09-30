"""Within-network connectivity boxplots and parcel connectomes."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns
from nilearn import plotting
from config import (
    GROUP_LABELS,
    NETWORK_ORDER,
    comparison_dir,
    conn_labels,
    derivpath,
    group_palette,
    load_parcels,
)


def main():
    df_within_net_conn = pd.read_csv(
        comparison_dir / "within_network_connectivity" / "subject_metrics.csv"
    )
    within_contrasts_all = pd.read_csv(
        comparison_dir / "within_network_connectivity" / "GEE_contrasts.csv"
    )
    pos = load_parcels()
    NETWORK_ORDER = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]
    BAND_ORDER = ["Theta", "Alpha", "Beta", "Gamma"]
    GROUP_ORDER = ["NCBP_trans", "NCBP_nontrans", "HC"]
    group_palette = {
        "NCBP_trans": "#D55E00",
        "NCBP_nontrans": "#009E73",
        "HC": "#0072B2",
    }
    conn_labels = {"aec": "AEC", "dwpli": "dWPLI"}
    within_figure_dir = Path(comparison_dir) / "within_network_connectivity" / "FIGURES"
    within_figure_dir.mkdir(parents=True, exist_ok=True)
    within_plot_data = df_within_net_conn.loc[
        df_within_net_conn["analysis_group"].isin(GROUP_ORDER)
        & df_within_net_conn["network"].isin(NETWORK_ORDER)
        & df_within_net_conn["freq_band"].isin(BAND_ORDER)
    ].copy()
    for column in ["mean_within_connectivity", "age"]:
        within_plot_data[column] = pd.to_numeric(
            within_plot_data[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
    within_plot_data = within_plot_data.dropna(
        subset=["sample_id", "analysis_group", "age", "sex", "mean_within_connectivity"]
    )

    def plot_within_con_box(con_type):
        df_sub = within_plot_data.loc[within_plot_data["con_type"].eq(con_type)]
        if df_sub.empty:
            return
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
        for ax, band in zip(axes.flat, BAND_ORDER):
            temp = df_sub.loc[df_sub["freq_band"].eq(band)]
            if temp.empty:
                ax.set_visible(False)
                continue
            sns.boxplot(
                data=temp,
                x="network",
                y="mean_within_connectivity",
                hue="analysis_group",
                order=NETWORK_ORDER,
                hue_order=GROUP_ORDER,
                palette=group_palette,
                dodge=True,
                width=0.8,
                gap=0.1,
                saturation=1,
                whis=1.5,
                showfliers=False,
                linewidth=1,
                boxprops={"edgecolor": "black", "alpha": 0.8},
                medianprops={"color": "black", "linewidth": 1.6},
                whiskerprops={"color": "black", "linewidth": 1},
                capprops={"color": "black", "linewidth": 1},
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=temp,
                x="network",
                y="mean_within_connectivity",
                hue="analysis_group",
                order=NETWORK_ORDER,
                hue_order=GROUP_ORDER,
                palette=group_palette,
                dodge=True,
                jitter=0.15,
                size=3.5,
                alpha=0.7,
                edgecolor="none",
                linewidth=0,
                zorder=4,
                legend=False,
                ax=ax,
            )
            ax.set_xlabel("Network", fontsize=18)
            ax.set_title(
                f"{conn_labels[con_type]} · {band} · Within-network connectivity",
                fontsize=20,
                fontweight="bold",
                pad=12,
            )
            ax.set_ylabel(f"Mean {conn_labels[con_type]}", fontsize=18)
            ax.tick_params(axis="both", labelsize=16)
            plt.setp(
                ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
            )
            ax.set_axisbelow(True)
            ax.xaxis.grid(False)
            ax.yaxis.grid(True, color="0.9", linewidth=0.8)
            sns.despine(ax=ax)
        handles = [
            Patch(
                facecolor=group_palette[group],
                edgecolor="black",
                label=GROUP_LABELS[group],
            )
            for group in GROUP_ORDER
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=3,
            frameon=False,
            fontsize=20,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.89], h_pad=2, w_pad=2)
        fig.savefig(
            within_figure_dir
            / f"network_within_connectivity_three_groups_{con_type}.svg",
            dpi=600,
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)

    for con_type in ["aec", "dwpli"]:
        plot_within_con_box(con_type)

    def plot_within_network_parcel_difference(
        first, second, con_type, band, network, edge_percentile=80
    ):
        parcel_info = pos.loc[
            pos["ROI Name"].astype(str).str.split("_").str[2].eq(network)
        ]
        parcel_labels = parcel_info["ROI Name"].astype(str).tolist()
        display_labels = [name.removeprefix("7Networks_") for name in parcel_labels]
        node_coords = parcel_info[["R", "A", "S"]].to_numpy(dtype=float)
        selected = within_plot_data.loc[
            within_plot_data["con_type"].eq(con_type)
            & within_plot_data["freq_band"].eq(band)
            & within_plot_data["network"].eq(network)
        ]

        def group_mean_matrix(analysis_group):
            participants = selected.loc[
                selected["analysis_group"].eq(analysis_group), ["group", "subject"]
            ].drop_duplicates()
            if participants.empty:
                raise ValueError(
                    f"No eligible participants for {analysis_group}, {con_type}, {band}, {network}."
                )
            total = np.zeros((len(parcel_labels), len(parcel_labels)), dtype=float)
            for source_group, subject in participants.itertuples(
                index=False, name=None
            ):
                file_path = (
                    Path(derivpath[source_group])
                    / subject
                    / f"{subject}_{con_type}_matrix_{band}.csv"
                )
                matrix = (
                    pd.read_csv(file_path, index_col=0)
                    .loc[parcel_labels, parcel_labels]
                    .to_numpy(dtype=float)
                )
                if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
                    raise ValueError(
                        f"Expected a finite, symmetric matrix: {file_path}"
                    )
                total += matrix
            return (total / len(participants), len(participants))

        first_matrix, n_first = group_mean_matrix(first)
        second_matrix, n_second = group_mean_matrix(second)
        difference = first_matrix - second_matrix
        np.fill_diagonal(difference, 0)
        colour_limit = float(np.max(np.abs(difference))) or 1.0
        norm = TwoSlopeNorm(vmin=-colour_limit, vcenter=0, vmax=colour_limit)
        fig = plt.figure(figsize=(17, 9), facecolor="white")
        fig.suptitle(
            f"{GROUP_LABELS[first]} vs {GROUP_LABELS[second]}",
            fontsize=22,
            fontweight="bold",
            y=0.96,
        )
        fig.text(
            0.5,
            0.89,
            f"{conn_labels[con_type]} · {band} · {network} network",
            ha="center",
            fontsize=20,
        )
        heat_ax = fig.add_axes([0.13, 0.34, 0.32, 0.47])
        brain_ax = fig.add_axes([0.51, 0.34, 0.46, 0.47])
        sns.heatmap(
            difference,
            mask=np.triu(np.ones_like(difference, dtype=bool)),
            cmap="turbo",
            norm=norm,
            square=True,
            linewidths=0.7,
            linecolor="white",
            xticklabels=display_labels,
            yticklabels=display_labels,
            cbar=False,
            ax=heat_ax,
        )
        plt.setp(
            heat_ax.get_xticklabels(),
            rotation=90,
            ha="right",
            rotation_mode="anchor",
            fontsize=16,
        )
        plt.setp(heat_ax.get_yticklabels(), rotation=0, fontsize=16)
        heat_ax.set_xlabel("Parcel", fontsize=20)
        heat_ax.set_ylabel("Parcel", fontsize=20)
        if edge_percentile is None:
            edge_threshold = None
        else:
            edge_threshold = f"{edge_percentile:g}%"
        plotting.plot_connectome(
            difference,
            node_coords,
            display_mode="lyrz",
            edge_threshold=edge_threshold,
            edge_cmap="turbo",
            edge_vmin=-colour_limit,
            edge_vmax=colour_limit,
            node_color="#F4A261",
            node_size=45,
            node_kwargs={"edgecolors": "#303030", "linewidths": 0.7},
            colorbar=False,
            annotate=False,
            black_bg=False,
            axes=brain_ax,
        )
        scalar_mappable = ScalarMappable(norm=norm, cmap="turbo")
        scalar_mappable.set_array([])
        cax = fig.add_axes([0.55, 0.22, 0.4, 0.025])
        cbar = fig.colorbar(scalar_mappable, cax=cax, orientation="horizontal")
        cbar.set_label(f"Mean {conn_labels[con_type]} difference", fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        fig.savefig(
            within_figure_dir
            / f"{first}_vs_{second}_{network}_{band}_{con_type}_parcel_connectome.svg",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    selected_results = within_contrasts_all.loc[
        within_contrasts_all["metric"].eq("mean_within_connectivity")
        & within_contrasts_all["passes_interaction_gate"].eq(True),
        ["first_group", "second_group", "con_type", "band", "network"],
    ].drop_duplicates()
    for row in selected_results.itertuples(index=False):
        plot_within_network_parcel_difference(
            first=row.first_group,
            second=row.second_group,
            con_type=row.con_type,
            band=row.band,
            network=row.network,
            edge_percentile=80,
        )


if __name__ == "__main__":
    main()
