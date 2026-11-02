"""Between-network connectivity boxplots, vertical adjusted pointplots and heatmaps."""

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns
from config import (
    GROUP_LABELS,
    NETWORK_ORDER,
    plot_results_dir, figure_dir,
    comparison_pairs,
    conn_labels,
    PLOT_GROUPS,
    group_palette, con_types, freq_bands,
)
from plotting_utils import (
    save_plot, add_significance_brackets, SHOW_SIGNIFICANCE,
    SIGNIFICANCE_COLOR, SIGNIFICANCE_LINEWIDTH, SIGNIFICANCE_FONTSIZE,
)


def plot_between_points(estimates, contrasts, con, band, out_dir):
    pairs = [f"{a}__{b}" for a, b in combinations(NETWORK_ORDER, 2)]
    means = estimates.loc[estimates.con_type.eq(con) & estimates.band.eq(band)]
    supported = contrasts.loc[
        contrasts.con_type.eq(con) & contrasts.band.eq(band)
        & contrasts.passes_interaction_gate.astype(str).str.lower().eq("true")
    ]
    offsets = dict(zip(PLOT_GROUPS, [-.23, 0, .23]))
    markers = {"NCBP_trans": "o", "NCBP_nontrans": "s", "HC": "^"}
    fig, ax = plt.subplots(figsize=(9, 13))
    for i in range(len(pairs)):
        if i % 2 == 0:
            ax.axhspan(i-.5, i+.5, color="#f2f4f6", zorder=0)
    for group in PLOT_GROUPS:
        rows = means.loc[means.analysis_group.eq(group)].set_index("network_pair").loc[pairs]
        y = np.arange(len(pairs)) + offsets[group]
        ax.errorbar(rows.adjusted_mean, y,
                    xerr=np.vstack([rows.adjusted_mean-rows.ci_low, rows.ci_high-rows.adjusted_mean]),
                    fmt=markers[group], linestyle="none", color=group_palette[group],
                    markersize=6, capsize=3, elinewidth=1.4, zorder=4)
    low, high = means.ci_low.min(), means.ci_high.max()
    step = max(high-low, .001)*.10
    for i, pair in enumerate(pairs):
        rows = supported.loc[supported.network_pair.eq(pair)] if SHOW_SIGNIFICANCE else supported.iloc[:0]
        for row in rows.itertuples():
            level = comparison_pairs.index((row.first_group, row.second_group))
            first, second = i+offsets[row.first_group], i+offsets[row.second_group]
            x = high + (level+1)*step
            mid = (first+second)/2
            ax.plot([x-step*.2, x, x], [first, first, mid], color=SIGNIFICANCE_COLOR, lw=SIGNIFICANCE_LINEWIDTH)
            ax.plot([x, x, x-step*.2], [mid, second, second], color=SIGNIFICANCE_COLOR, lw=SIGNIFICANCE_LINEWIDTH)
            ax.text(x+step*.1, mid, "*", ha="left", va="center", fontsize=SIGNIFICANCE_FONTSIZE, color=SIGNIFICANCE_COLOR)
    ax.set(ylim=(len(pairs)-.4, -.6),
           xlim=(low-step, high+step*(4.5 if SHOW_SIGNIFICANCE and len(supported) else 1)))
    ax.set_yticks(range(len(pairs)), [pair.replace("__", " – ") for pair in pairs])
    ax.tick_params(labelsize=13)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel(f"Adjusted mean {conn_labels[con]} (95% CI)", fontsize=15, labelpad=12)
    ax.set_ylabel("")
    ax.set_title(f"{conn_labels[con]} · {band}", fontsize=18, pad=16)
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    sns.despine(ax=ax, left=True)
    fig.legend(handles=[Line2D([], [], color=group_palette[g], marker=markers[g],
                              linestyle="none", markersize=9, label=GROUP_LABELS[g]) for g in PLOT_GROUPS],
               loc="upper center", bbox_to_anchor=(.5, .995), ncol=3, frameon=False, fontsize=12,
               columnspacing=1.1, handletextpad=.4)
    fig.subplots_adjust(left=.31, right=.97, bottom=.075, top=.92)
    stem = f"network_between_connectivity_{con}_{band}_adjusted_pointplot"
    save_plot(fig, out_dir / stem)


def main(boxes_only=False):
    plt.rcParams["svg.fonttype"] = "none"
    df_between_net_conn = pd.read_csv(
        plot_results_dir / "between_network_connectivity" / "subject_metrics.csv"
    )
    between_contrasts_all = pd.read_csv(
        plot_results_dir / "between_network_connectivity" / "GEE_contrasts.csv"
    )
    adjusted_means = pd.read_csv(plot_results_dir / "between_network_connectivity" / "GEE_adjusted_means.csv")
    network_figure_dir = figure_dir / "between_network_connectivity" / "FIGURES"
    network_figure_dir.mkdir(parents=True, exist_ok=True)
    if not boxes_only:
        for con in con_types:
            for band in freq_bands:
                plot_between_points(adjusted_means, between_contrasts_all,
                                    con, band, network_figure_dir)
    BETWEEN_PAIR_ORDER = [f"{a}__{b}" for a, b in combinations(NETWORK_ORDER, 2)]
    BETWEEN_GROUP_ORDER = PLOT_GROUPS
    BETWEEN_GROUP_PALETTE = group_palette
    between_boxplot_dir = network_figure_dir

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
            boxprops={"edgecolor": "black", "linewidth": 1.0, "alpha": .8},
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
            alpha=1.0,
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
        rows = between_contrasts_all.loc[
            between_contrasts_all.con_type.eq(con_type)
            & between_contrasts_all.band.eq(band)
        ]
        offsets = dict(zip(BETWEEN_GROUP_ORDER, [-.8/3, 0, .8/3]))
        add_significance_brackets(ax, rows, BETWEEN_PAIR_ORDER, "network_pair",
                                  offsets, show=SHOW_SIGNIFICANCE)
        save_plot(fig, out_file)

    for con_type in con_types:
        for band in freq_bands:
            plot_between_connectivity_boxplot(
                df=df_between_net_conn,
                con_type=con_type,
                band=band,
                out_file=between_boxplot_dir
                / f"network_between_connectivity_{con_type}_{band}_boxplot.svg",
            )
    if boxes_only:
        return
    BAND_ORDER = list(freq_bands)
    CMAP = plt.get_cmap("turbo")
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

    def plot_between_network(first, second, con_type):
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
        scalar_mappable = ScalarMappable(norm=norm, cmap=CMAP)
        scalar_mappable.set_array([])
        cbar = fig.colorbar(
            scalar_mappable, ax=list(axes.flat), shrink=0.8, fraction=0.025, pad=0.03
        )
        cbar.set_label("Adjusted $\\beta$", fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        save_plot(fig, network_figure_dir /
                  f"network_between_heatmap_{first}_vs_{second}_{con_type}")

    for first, second in comparison_pairs:
        for con_type in con_types:
            plot_between_network(first=first, second=second, con_type=con_type)


if __name__ == "__main__":
    main()
