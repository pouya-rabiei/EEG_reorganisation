"""Shared figure styling; statistical models remain in their outcome scripts."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from config import (
    PLOT_GROUPS, GROUP_LABELS, group_palette, conn_labels, freq_bands,
    WITHIN_NETWORKS, WITHIN_NETWORK_LABELS, NETWORK_ORDER,
)

SHOW_SIGNIFICANCE = True
SIGNIFICANCE_COLOR = "black"
SIGNIFICANCE_LINEWIDTH = 1.3
SIGNIFICANCE_FONTSIZE = 17


def add_significance_brackets(ax, contrasts, order, category, offsets, show=True):
    """Mark saved interaction-supported contrasts; no statistical tests are run."""
    if not show or contrasts is None or contrasts.empty:
        return
    supported = contrasts.loc[
        contrasts.passes_interaction_gate.astype(str).str.lower().eq("true")
    ]
    low, high = ax.dataLim.ymin, ax.dataLim.ymax
    step = max(high - low, abs(high) * .01, .001) * .14
    highest = high
    for i, name in enumerate(order):
        rows = supported.loc[supported[category].eq(name)]
        for level, row in enumerate(rows.itertuples(index=False)):
            first = i + offsets[row.first_group]
            second = i + offsets[row.second_group]
            mid = (first + second) / 2
            y = high + (level + 1) * step
            ax.plot([first, first, mid], [y-step*.2, y, y],
                    color=SIGNIFICANCE_COLOR, lw=SIGNIFICANCE_LINEWIDTH, zorder=6)
            ax.plot([mid, second, second], [y, y, y-step*.2],
                    color=SIGNIFICANCE_COLOR, lw=SIGNIFICANCE_LINEWIDTH, zorder=6)
            ax.text(mid, y, "*", ha="center", va="bottom", fontsize=SIGNIFICANCE_FONTSIZE,
                    color=SIGNIFICANCE_COLOR, zorder=6)
            highest = max(highest, y + step)
    if highest > high:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, max(top, highest))


def save_plot(fig, path, tight=True):
    for suffix in [".svg", ".png"]:
        fig.savefig(path.with_suffix(suffix), dpi=600,
                    bbox_inches="tight" if tight else None, facecolor="white")
    plt.close(fig)


def plot_network_boxes(data, con, metric, label, output_dir, contrasts=None,
                       show_significance=SHOW_SIGNIFICANCE):
    rows = None if contrasts is None else contrasts.loc[
        contrasts.con_type.eq(con) & contrasts.metric.eq(metric)
    ]
    networks = data.loc[data.con_type.eq(con), "network"].unique()
    network_order = [n for n in NETWORK_ORDER if n in networks]
    if metric in ("within_network_clustering", "within_network_efficiency"):
        network_order = [n for n in network_order if n != "Limbic"]
    selected = data.loc[data.con_type.eq(con) & data.network.isin(network_order)]
    selected = selected.dropna(subset=[metric, "age", "sex", "sample_id"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    np.random.seed(42)
    for ax, band in zip(axes.flat, freq_bands):
        subset = selected.loc[selected.freq_band.eq(band)]
        common = dict(data=subset, x="network", y=metric, hue="analysis_group",
                      order=network_order, hue_order=PLOT_GROUPS,
                      palette=group_palette, dodge=True, legend=False, ax=ax)
        sns.boxplot(
            **common, width=.8, gap=.1, saturation=1, whis=1.5,
            showfliers=False, linewidth=1,
            boxprops={"edgecolor": "black", "alpha": .8},
            medianprops={"color": "black", "linewidth": 1.6},
            whiskerprops={"color": "black"}, capprops={"color": "black"},
        )
        sns.stripplot(**common, jitter=.15, size=3.5, alpha=1.0,
                      edgecolor="none", linewidth=0, zorder=4)
        ax.set_xlabel("Network", fontsize=18)
        ax.set_ylabel(label, fontsize=18)
        ax.set_title(f"{conn_labels[con]} · {band} · {label}",
                     fontsize=20, fontweight="bold", pad=12)
        ax.tick_params(labelsize=16)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=".85", linewidth=.7)
        sns.despine(ax=ax)
        band_rows = None if rows is None else rows.loc[rows.band.eq(band)]
        offsets = dict(zip(PLOT_GROUPS, [-.8/3, 0, .8/3]))
        add_significance_brackets(ax, band_rows, network_order, "network",
                                  offsets, show=show_significance)
    handles = [Patch(facecolor=group_palette[g], label=GROUP_LABELS[g]) for g in PLOT_GROUPS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .975),
               ncol=3, frameon=False, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, .91], h_pad=2, w_pad=2)
    save_plot(fig, output_dir / f"network_three_groups_{con}_{metric}_boxplot")


def plot_rainclouds(data, estimates, contrasts, output_dir, metric_labels):
    keys = ["con_type", "band", "metric", "network"]
    locations = estimates[keys].drop_duplicates()
    rng = np.random.default_rng(42)
    markers = {"NCBP_trans": "o", "NCBP_nontrans": "s", "HC": "^"}
    for con, band, metric, network in locations.itertuples(index=False, name=None):
        observed = data.loc[
            data.con_type.eq(con) & data.freq_band.eq(band) & data.network.eq(network)
        ].replace([np.inf, -np.inf], np.nan).dropna(subset=[metric, "age", "sex", "sample_id"])
        means = estimates.loc[
            estimates.con_type.eq(con) & estimates.band.eq(band)
            & estimates.metric.eq(metric) & estimates.network.eq(network)
        ].set_index("analysis_group")
        supported = contrasts.loc[
            contrasts.con_type.eq(con) & contrasts.band.eq(band)
            & contrasts.metric.eq(metric) & contrasts.network.eq(network)
            & contrasts.passes_interaction_gate.eq(True)
        ]
        fig, ax = plt.subplots(figsize=(9, 6))
        for x, group in enumerate(PLOT_GROUPS):
            values = observed.loc[observed.analysis_group.eq(group), metric].to_numpy(float)
            color = group_palette[group]
            if len(values) > 1 and np.ptp(values) > 0:
                violin = ax.violinplot(values, positions=[x], widths=.72,
                                      showmeans=False, showmedians=False, showextrema=False)
                body = violin["bodies"][0]
                vertices = body.get_paths()[0].vertices
                vertices[:, 0] = np.minimum(vertices[:, 0], x)
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(.8)
            ax.scatter(x + rng.uniform(.15, .36, len(values)), values,
                       s=22, color=color, alpha=1.0, edgecolors="none", zorder=3)
            row = means.loc[group]
            ax.errorbar(x + .06, row.adjusted_mean,
                        yerr=[[row.adjusted_mean-row.ci_low], [row.ci_high-row.adjusted_mean]],
                        fmt=markers[group], color=color, mec="black", mew=1,
                        markersize=10, capsize=5, elinewidth=2, zorder=5)
        low = min(observed[metric].min(), means.ci_low.min())
        high = max(observed[metric].max(), means.ci_high.max())
        step = max(high-low, .05) * .10
        for level, row in enumerate(supported.itertuples()):
            first = PLOT_GROUPS.index(row.first_group) + .06
            second = PLOT_GROUPS.index(row.second_group) + .06
            y = high + (level+1)*step
            mid = (first+second)/2
            ax.plot([first, first, mid], [y-step*.2, y, y], color=SIGNIFICANCE_COLOR, lw=1.5)
            ax.plot([mid, second, second], [y, y, y-step*.2], color=SIGNIFICANCE_COLOR, lw=1.5)
            ax.text(mid, y, "*", ha="center", va="bottom", fontsize=19, color=SIGNIFICANCE_COLOR)
        ax.set_ylim(low-step*.4, high+step*(len(supported)+1.2))
        ax.set_xticks(range(3), [GROUP_LABELS[g].replace(" ", "\n") for g in PLOT_GROUPS])
        ax.set_ylabel(metric_labels[metric], fontsize=18)
        ax.set_title(f"{conn_labels[con]} · {band} · {metric_labels[metric]}\n"
                     f"{WITHIN_NETWORK_LABELS[network]}", fontsize=19, pad=15)
        ax.tick_params(labelsize=15)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=".9")
        sns.despine(ax=ax)
        fig.tight_layout()
        save_plot(fig, output_dir / f"raincloud_{con}_{band}_{metric}_{network}")
