"""Whole-brain clustering and efficiency distributions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config import GROUP_LABELS, plot_results_dir, figure_dir, conn_labels, group_palette, PLOT_GROUPS, freq_bands
from plotting_utils import add_significance_brackets, SHOW_SIGNIFICANCE


def main():
    output_dir = plot_results_dir / "whole_brain_topology"
    df_net_metric_groups = pd.read_csv(
        output_dir / "whole_brain_metrics.csv"
    )
    freq_order = list(freq_bands)
    conn_order = ["aec", "dwpli"]
    contrasts = pd.concat([
        pd.read_csv(output_dir / f"whole_brain_{con}_GEE_band_contrasts.csv")
        for con in conn_order
    ], ignore_index=True)
    group_order = PLOT_GROUPS
    group_markers = {"NCBP_trans": "o", "NCBP_nontrans": "s", "HC": "^"}
    metric_ylabels = {
        "global_clustering": "Clustering coefficient (unitless)",
        "global_efficiency": "Efficiency (unitless)",
    }
    metrics_whole_brain = list(metric_ylabels)
    metric_titles = {
        metric: metric.replace('_', ' ').capitalize() for metric in metrics_whole_brain
    }
    whole_brain_plot_dir = figure_dir / "whole_brain_topology" / "FIGURES"
    whole_brain_plot_dir.mkdir(parents=True, exist_ok=True)
    df_whole_brain_plot = df_net_metric_groups.copy()
    df_whole_brain_plot = df_whole_brain_plot.loc[
        df_whole_brain_plot["analysis_group"].isin(group_order)
        & df_whole_brain_plot["freq_band"].isin(freq_order)
        & df_whole_brain_plot["con_type"].isin(conn_order)
    ].copy()
    for column in metrics_whole_brain + ["age"]:
        df_whole_brain_plot[column] = pd.to_numeric(
            df_whole_brain_plot[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
    df_whole_brain_plot["freq_band"] = pd.Categorical(
        df_whole_brain_plot["freq_band"], categories=freq_order, ordered=True
    )

    def metric_plot_data(con_type, metric):
        data = (
            df_whole_brain_plot.loc[df_whole_brain_plot["con_type"].eq(con_type)]
            .dropna(
                subset=[
                    "sample_id",
                    "analysis_group",
                    "freq_band",
                    "age",
                    "sex",
                    metric,
                ]
            )
            .copy()
        )
        if data.duplicated(["sample_id", "freq_band"]).any():
            raise ValueError(f"Duplicate participant-band rows: {con_type} | {metric}")
        return data

    def draw_violin_box(
        ax, values, position, color, rng, violin_width=0.24, box_width=0.075
    ):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        if values.size > 1 and np.ptp(values) > 0:
            violin = ax.violinplot(
                dataset=[values],
                positions=[position],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(.65)
        jitter = rng.uniform(
            -violin_width * 0.15, violin_width * 0.15, size=values.size
        )
        ax.scatter(
            position + jitter,
            values,
            s=15,
            color=color,
            alpha=1.0,
            edgecolors="none",
            zorder=3,
        )
        ax.boxplot(
            [values],
            positions=[position],
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            whis=1.5,
            zorder=4,
            boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.2},
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
        )

    def style_metric_axis(ax, metric):
        ax.set_ylabel(metric_ylabels[metric], fontsize=20, labelpad=10)
        ax.tick_params(axis="both", labelsize=18)
        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.9", linewidth=0.7)
        ax.margins(y=0.12)

    def save_whole_brain_figure(fig, filename):
        fig.savefig(
            whole_brain_plot_dir / f"{filename}.svg",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(whole_brain_plot_dir / f"{filename}.png", dpi=600,
                    bbox_inches="tight", facecolor="white")

    with plt.rc_context(
        {
            "font.size": 20,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    ):
        for con_type in conn_order:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.8, wspace=0.3)
            rng = np.random.default_rng(42)
            for ax, metric in zip(axes, metrics_whole_brain):
                data = metric_plot_data(con_type, metric)
                subject_average = data.groupby(
                    ["sample_id", "analysis_group"], observed=True, as_index=False
                ).agg(value=(metric, "mean"), n_bands=("freq_band", "nunique"))
                subject_average = subject_average.loc[
                    subject_average["n_bands"].eq(len(freq_order))
                ]
                for position, group in enumerate(group_order):
                    values = subject_average.loc[
                        subject_average["analysis_group"].eq(group), "value"
                    ].to_numpy()
                    draw_violin_box(
                        ax=ax,
                        values=values,
                        position=position,
                        color=group_palette[group],
                        rng=rng,
                        violin_width=0.65,
                        box_width=0.18,
                    )
                ax.set_xticks(np.arange(len(group_order)))
                ax.set_xticklabels(
                    [GROUP_LABELS[group] for group in group_order], rotation=30
                )
                ax.set_xlim(-0.6, len(group_order) - 0.4)
                ax.set_xlabel("")
                style_metric_axis(ax, metric)
                ax.set_title(
                    f"{conn_labels[con_type]} · {metric_titles[metric]}\n"
                    "Mean across bands",
                    fontsize=20,
                    fontweight="bold",
                    pad=12,
                )
            save_whole_brain_figure(
                fig, f"whole_brain_{con_type}_three_groups_across_bands"
            )
            plt.close(fig)
        offsets = {"NCBP_trans": -0.27, "NCBP_nontrans": 0.0, "HC": 0.27}
        x_positions = np.arange(len(freq_order))
        for con_type in conn_order:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.subplots_adjust(
                left=0.075, right=0.98, bottom=0.16, top=0.73, wspace=0.28
            )
            rng = np.random.default_rng(42)
            for ax, metric in zip(axes, metrics_whole_brain):
                data = metric_plot_data(con_type, metric)
                for group in group_order:
                    group_data = data.loc[data["analysis_group"].eq(group)]
                    positions = x_positions + offsets[group]
                    for position, band in zip(positions, freq_order):
                        values = group_data.loc[
                            group_data["freq_band"].eq(band), metric
                        ].to_numpy()
                        draw_violin_box(
                            ax=ax,
                            values=values,
                            position=position,
                            color=group_palette[group],
                            rng=rng,
                        )
                    means = (
                        group_data.groupby("freq_band", observed=True)[metric]
                        .mean()
                        .reindex(freq_order)
                    )
                    ax.plot(
                        positions,
                        means.to_numpy(dtype=float),
                        color=group_palette[group],
                        marker=group_markers[group],
                        markersize=6,
                        linewidth=2.0,
                        zorder=5,
                    )
                ax.set_xticks(x_positions)
                ax.set_xticklabels(freq_order, rotation=30)
                ax.set_xlim(-0.6, len(freq_order) - 0.4)
                ax.set_xlabel("Frequency band", fontsize=20, labelpad=10)
                style_metric_axis(ax, metric)
                rows = contrasts.loc[
                    contrasts.con_type.eq(con_type) & contrasts.metric.eq(metric)
                ]
                add_significance_brackets(ax, rows, freq_order, "band", offsets,
                                          show=SHOW_SIGNIFICANCE)
                ax.set_title(
                    f"{conn_labels[con_type]} · {metric_titles[metric]}",
                    fontsize=20,
                    fontweight="bold",
                    pad=12,
                )
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color=group_palette[group],
                    marker=group_markers[group],
                    linewidth=2.0,
                    markersize=6,
                    label=GROUP_LABELS[group],
                )
                for group in group_order
            ]
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.53, 0.91),
                ncol=3,
                frameon=False,
                fontsize=20,
            )
            save_whole_brain_figure(
                fig, f"whole_brain_{con_type}_three_groups_bandwise_violin"
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
