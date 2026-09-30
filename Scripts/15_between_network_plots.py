"""Between-network connectivity boxplots heatmaps and circles."""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns
from config import (
    GROUP_LABELS,
    NETWORK_ORDER,
    comparison_dir,
    comparison_pairs,
    conn_labels,
)


def main():
    df_between_net_conn = pd.read_csv(
        comparison_dir / "between_network_connectivity" / "subject_metrics.csv"
    )
    between_contrasts_all = pd.read_csv(
        comparison_dir / "between_network_connectivity" / "GEE_contrasts.csv"
    )
    network_figure_dir = comparison_dir / "between_network_connectivity" / "FIGURES"
    network_figure_dir.mkdir(parents=True, exist_ok=True)
    BETWEEN_NETWORK_ORDER = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]
    BETWEEN_PAIR_ORDER = [
        f"{first}__{second}" for first, second in combinations(BETWEEN_NETWORK_ORDER, 2)
    ]
    BETWEEN_GROUP_ORDER = ["NCBP_trans", "NCBP_nontrans", "HC"]
    BETWEEN_GROUP_PALETTE = {
        "NCBP_trans": "#D55E00",
        "NCBP_nontrans": "#009E73",
        "HC": "#0072B2",
    }
    between_boxplot_dir = (
        Path(comparison_dir) / "between_network_connectivity" / "FIGURES"
    )

    def plot_between_connectivity_boxplot(df, con_type, band, out_file):
        value_col = "mean_between_connectivity"
        data = df.loc[
            df["con_type"].eq(con_type)
            & df["freq_band"].eq(band)
            & df["analysis_group"].isin(BETWEEN_GROUP_ORDER)
            & df["network_pair"].isin(BETWEEN_PAIR_ORDER)
        ].copy()
        for column in [value_col, "age"]:
            data[column] = pd.to_numeric(data[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        data = data.dropna(
            subset=[value_col, "age", "sex", "sample_id", "analysis_group"]
        )
        if data.empty:
            return
        fig, ax = plt.subplots(figsize=(28, 8))
        fig.subplots_adjust(left=0.055, right=0.995, bottom=0.34, top=0.79)
        sns.boxplot(
            data=data,
            x="network_pair",
            y=value_col,
            hue="analysis_group",
            order=BETWEEN_PAIR_ORDER,
            hue_order=BETWEEN_GROUP_ORDER,
            palette=BETWEEN_GROUP_PALETTE,
            dodge=True,
            width=0.8,
            gap=0.1,
            saturation=1,
            whis=1.5,
            showfliers=False,
            linewidth=1.0,
            boxprops={"edgecolor": "black", "linewidth": 1.0, "alpha": 0.8},
            medianprops={"color": "black", "linewidth": 1.6},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=data,
            x="network_pair",
            y=value_col,
            hue="analysis_group",
            order=BETWEEN_PAIR_ORDER,
            hue_order=BETWEEN_GROUP_ORDER,
            palette=BETWEEN_GROUP_PALETTE,
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
        ax.set_xticks(np.arange(len(BETWEEN_PAIR_ORDER)))
        ax.set_xticklabels(
            [pair.replace("__", "–") for pair in BETWEEN_PAIR_ORDER],
            rotation=30,
            ha="right",
            rotation_mode="anchor",
            fontsize=16,
        )
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xlabel("Network pair", fontsize=18, labelpad=12)
        ax.set_ylabel("Mean between-network connectivity", fontsize=18)
        ax.set_title(
            f"{conn_labels[con_type]} · {band} · Between-network connectivity",
            fontsize=20,
            fontweight="bold",
            pad=12,
        )
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color="0.85", linewidth=0.7)
        sns.despine(ax=ax)
        fig.legend(
            handles=[
                Patch(facecolor=BETWEEN_GROUP_PALETTE[group], label=GROUP_LABELS[group])
                for group in BETWEEN_GROUP_ORDER
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=3,
            frameon=False,
            fontsize=20,
        )
        fig.savefig(out_file, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(fig)

    for con_type in ["aec", "dwpli"]:
        for band in ["Theta", "Alpha", "Beta", "Gamma"]:
            plot_between_connectivity_boxplot(
                df=df_between_net_conn,
                con_type=con_type,
                band=band,
                out_file=between_boxplot_dir
                / f"network_between_connectivity_{con_type}_{band}_boxplot.svg",
            )
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
    comparison_pairs = [
        ("NCBP_trans", "NCBP_nontrans"),
        ("NCBP_trans", "HC"),
        ("NCBP_nontrans", "HC"),
    ]
    network_color_map = {
        "Vis": "#1F77B4",
        "SomMot": "#FF7F0E",
        "DorsAttn": "#2CA02C",
        "SalVentAttn": "#D62728",
        "Limbic": "#9467BD",
        "Cont": "#8C564B",
        "Default": "#E377C2",
    }
    CMAP = plt.get_cmap("turbo")
    SIG_THRESHOLD = 0.05
    df_between_plot = between_contrasts_all.loc[
        between_contrasts_all["metric"].eq("mean_between_connectivity")
    ].copy()
    df_between_plot[["network_a", "network_b"]] = (
        df_between_plot["network_pair"].astype(str).str.split("__", expand=True)
    )
    df_between_plot["plot_significant"] = df_between_plot["passes_interaction_gate"].eq(
        True
    )
    colour_limits = (
        df_between_plot.groupby("con_type", observed=True)["beta"]
        .agg(lambda values: values.abs().max())
        .to_dict()
    )

    def make_between_matrices(rows):
        n_networks = len(NETWORK_ORDER)
        matrix = np.full((n_networks, n_networks), np.nan)
        significant = np.zeros((n_networks, n_networks), dtype=bool)
        network_indices = {
            network: index for index, network in enumerate(NETWORK_ORDER)
        }
        for row in rows.itertuples(index=False):
            i = network_indices[row.network_a]
            j = network_indices[row.network_b]
            matrix[i, j] = matrix[j, i] = row.beta
            significant[i, j] = significant[j, i] = row.plot_significant
        return (matrix, significant)

    sector_gap = np.deg2rad(6)
    sector_width = (2 * np.pi - len(NETWORK_ORDER) * sector_gap) / len(NETWORK_ORDER)
    sector_angles = {
        network: (
            index * (sector_width + sector_gap),
            index * (sector_width + sector_gap) + sector_width,
        )
        for index, network in enumerate(NETWORK_ORDER)
    }
    sector_centres = {
        network: (start + end) / 2 for network, (start, end) in sector_angles.items()
    }

    def connection_curve(theta1, theta2):
        first = np.array([np.sin(theta1), np.cos(theta1)])
        second = np.array([np.sin(theta2), np.cos(theta2)])
        control = 0.28 * (first + second)
        t = np.linspace(0, 1, 180)[:, None]
        curve = (1 - t) ** 2 * first + 2 * (1 - t) * t * control + t**2 * second
        return (curve[:, 0], curve[:, 1])

    def plot_between_network(first, second, con_type, plot_type):
        selected = df_between_plot.loc[
            df_between_plot["first_group"].eq(first)
            & df_between_plot["second_group"].eq(second)
            & df_between_plot["con_type"].eq(con_type)
        ]
        if selected.empty:
            return
        limit = float(colour_limits[con_type])
        if not np.isfinite(limit) or limit == 0:
            limit = 1.0
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
        fig.suptitle(
            f"{GROUP_LABELS[first]} vs {GROUP_LABELS[second]}",
            fontsize=22,
            fontweight="bold",
        )
        for ax, band in zip(axes.flat, BAND_ORDER):
            rows = selected.loc[selected["band"].eq(band)]
            if rows.empty:
                ax.set_axis_off()
                continue
            ax.set_title(
                f"{conn_labels[con_type]} · {band}",
                fontsize=20,
                fontweight="bold",
                pad=14,
            )
            if plot_type == "heatmap":
                matrix, significant = make_between_matrices(rows)
                lower_triangle = np.tril(np.ones(matrix.shape, dtype=bool), k=-1)
                highlighted = significant & lower_triangle
                sns.heatmap(
                    matrix,
                    mask=np.isnan(matrix),
                    cmap=CMAP,
                    vmin=-limit,
                    vmax=limit,
                    square=True,
                    linewidths=0.5,
                    linecolor="white",
                    cbar=False,
                    alpha=0.18,
                    xticklabels=NETWORK_ORDER,
                    yticklabels=NETWORK_ORDER,
                    ax=ax,
                )
                if highlighted.any():
                    sns.heatmap(
                        matrix,
                        mask=~highlighted,
                        cmap=CMAP,
                        vmin=-limit,
                        vmax=limit,
                        square=True,
                        linewidths=0.5,
                        linecolor="white",
                        cbar=False,
                        xticklabels=NETWORK_ORDER,
                        yticklabels=NETWORK_ORDER,
                        ax=ax,
                    )
                for i, j in zip(*np.where(highlighted)):
                    beta_value = matrix[i, j]
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{beta_value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="white" if abs(beta_value) > 0.65 * limit else "black",
                    )
                ax.set_xticklabels(
                    NETWORK_ORDER,
                    rotation=30,
                    ha="right",
                    rotation_mode="anchor",
                    fontsize=18,
                )
                ax.set_yticklabels(NETWORK_ORDER, rotation=0, fontsize=18)
                ax.set_xlabel("Network", fontsize=20)
                ax.set_ylabel("Network", fontsize=20)
            else:
                ax.set_aspect("equal")
                ax.set_xlim(-1.45, 1.45)
                ax.set_ylim(-1.35, 1.35)
                ax.set_axis_off()
                for network in NETWORK_ORDER:
                    start, end = sector_angles[network]
                    theta = np.linspace(start, end, 180)
                    centre = sector_centres[network]
                    ax.plot(
                        np.sin(theta),
                        np.cos(theta),
                        color=network_color_map[network],
                        linewidth=3,
                        zorder=3,
                    )
                    ax.text(
                        1.16 * np.sin(centre),
                        1.16 * np.cos(centre),
                        network,
                        ha="center",
                        va="center",
                        fontsize=18,
                        fontweight="bold",
                        color=network_color_map[network],
                    )
                significant_rows = rows.loc[rows["plot_significant"]]
                if significant_rows.empty:
                    message = (
                        "No corrected pairwise\ncontrasts"
                        if rows["interaction_p_corrected"].lt(SIG_THRESHOLD).any()
                        else "Interaction not\nsignificant"
                    )
                    ax.text(
                        0,
                        0,
                        message,
                        ha="center",
                        va="center",
                        fontsize=16,
                        color="0.4",
                    )
                ordered_indices = significant_rows["beta"].abs().sort_values().index
                for row in significant_rows.loc[ordered_indices].itertuples(
                    index=False
                ):
                    x, y = connection_curve(
                        sector_centres[row.network_a], sector_centres[row.network_b]
                    )
                    line_width = 1.5 + 5.5 * abs(row.beta) / limit
                    ax.plot(
                        x,
                        y,
                        color=CMAP(norm(row.beta)),
                        linewidth=line_width,
                        alpha=1,
                        zorder=2,
                    )
        scalar_mappable = ScalarMappable(norm=norm, cmap=CMAP)
        scalar_mappable.set_array([])
        cbar = fig.colorbar(
            scalar_mappable, ax=list(axes.flat), shrink=0.8, fraction=0.025, pad=0.03
        )
        cbar.set_label("Adjusted $\\beta$", fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        fig.savefig(
            network_figure_dir
            / f"network_between_{plot_type}_{first}_vs_{second}_{con_type}.svg",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    for first, second in comparison_pairs:
        for con_type in ["aec", "dwpli"]:
            for plot_type in ["heatmap", "circos"]:
                plot_between_network(
                    first=first, second=second, con_type=con_type, plot_type=plot_type
                )


if __name__ == "__main__":
    main()
