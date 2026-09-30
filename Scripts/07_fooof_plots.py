"""Spectral parameter boxplots and glass-brain maps."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.colors import Normalize, TwoSlopeNorm
import nibabel as nib
from nilearn import datasets, plotting
from config import GROUP_LABELS, NETWORK_ORDER, PLOT_GROUPS, group_palette, spectral_dir


def main():
    fooof_net_all = pd.read_csv(spectral_dir / "fooof_network_summary_all.csv")
    fooof_contrasts_df = pd.read_csv(spectral_dir / "FOOOF_network_GEE_contrasts.csv")
    fooof_roi_plot = pd.read_csv(spectral_dir / "fooof_roi_all.csv")
    fooof_glass_dir = spectral_dir / "FOOOF_GLASS_BRAIN"
    fooof_roi_adjusted_means_df = pd.read_csv(
        fooof_glass_dir / "FOOOF_ROI_adjusted_means.csv"
    )
    fooof_roi_adjusted_effects_df = pd.read_csv(
        fooof_glass_dir / "FOOOF_ROI_adjusted_effect_sizes_FDR.csv"
    )
    fooof_fig_dir = Path(spectral_dir) / "FOOOF_BOXPLOTS"
    fooof_fig_dir.mkdir(parents=True, exist_ok=True)
    NETWORK_ORDER = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]
    GROUP_LABELS = {
        "NCBP_trans": "Transitioners",
        "NCBP_nontrans": "Non-Transitioners",
        "HC": "Healthy Controls",
    }
    PLOT_GROUPS = ["NCBP_trans", "NCBP_nontrans", "HC"]
    METRIC_LABELS = {
        "exponent_mean": "Exponent",
        "offset_mean": "Offset",
        "alpha_pw_mean": "Alpha-peak power",
    }
    SHOW_POINTS = True
    plot_data = fooof_net_all.copy()
    contrast_data = fooof_contrasts_df.copy()
    contrast_data["p_corrected"] = pd.to_numeric(
        contrast_data["p_corrected"], errors="coerce"
    )
    group_offsets = {"NCBP_trans": -0.25, "NCBP_nontrans": 0.0, "HC": 0.25}
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
        for metric, metric_label in METRIC_LABELS.items():
            df = plot_data.loc[
                plot_data["analysis_group"].isin(PLOT_GROUPS)
                & plot_data["network"].isin(NETWORK_ORDER)
            ].copy()
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
            df = df.dropna(
                subset=["sample_id", "analysis_group", "network", "age", "sex", metric]
            )
            if df.empty:
                continue
            if df.duplicated(["sample_id", "network"]).any():
                raise ValueError(f"Duplicate participant-network rows for {metric}")
            fig, ax = plt.subplots(figsize=(15, 6.5))
            fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.8)
            rng = np.random.default_rng(42)
            y_min = float(df[metric].min())
            y_max = float(df[metric].max())
            y_span = y_max - y_min
            if y_span == 0:
                y_span = max(abs(y_max), 1.0) * 0.1
            highest_annotation = y_max
            for network_index, network in enumerate(NETWORK_ORDER):
                network_df = df.loc[df["network"].eq(network)]
                if network_df.empty:
                    continue
                for group in PLOT_GROUPS:
                    values = network_df.loc[
                        network_df["analysis_group"].eq(group), metric
                    ].to_numpy(dtype=float)
                    if values.size == 0:
                        continue
                    x_position = network_index + group_offsets[group]
                    ax.boxplot(
                        [values],
                        positions=[x_position],
                        widths=0.21,
                        patch_artist=True,
                        manage_ticks=False,
                        whis=1.5,
                        showfliers=not SHOW_POINTS,
                        boxprops={
                            "facecolor": group_palette[group],
                            "edgecolor": "black",
                            "linewidth": 1.0,
                            "alpha": 0.8,
                        },
                        medianprops={"color": "black", "linewidth": 1.6},
                        whiskerprops={"color": "black", "linewidth": 1.0},
                        capprops={"color": "black", "linewidth": 1.0},
                        flierprops={
                            "marker": "o",
                            "markersize": 3,
                            "markerfacecolor": group_palette[group],
                            "markeredgecolor": "none",
                            "alpha": 0.5,
                        },
                    )
                    if SHOW_POINTS:
                        jitter = rng.uniform(-0.065, 0.065, size=values.size)
                        ax.scatter(
                            x_position + jitter,
                            values,
                            s=12,
                            color=group_palette[group],
                            edgecolors="none",
                            alpha=0.7,
                            zorder=4,
                        )
            ax.set_xticks(np.arange(len(NETWORK_ORDER)))
            ax.set_xticklabels(
                NETWORK_ORDER,
                fontsize=20,
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
            ax.set_xlabel("Network", fontsize=22, labelpad=10)
            ax.set_ylabel(metric_label, fontsize=22, labelpad=10)
            fig.suptitle(metric_label, fontsize=22, fontweight="bold", y=1.01)
            ax.set_xlim(-0.6, len(NETWORK_ORDER) - 0.4)
            ax.set_ylim(y_min - 0.07 * y_span, highest_annotation + 0.08 * y_span)
            ax.tick_params(axis="y", labelsize=20)
            ax.tick_params(axis="x", length=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="0.9", linewidth=0.8)
            legend_handles = [
                Patch(
                    facecolor=group_palette[group],
                    edgecolor="black",
                    label=GROUP_LABELS[group],
                )
                for group in PLOT_GROUPS
            ]
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.53, 0.93),
                ncol=3,
                frameon=False,
                fontsize=20,
            )
            fig.savefig(
                fooof_fig_dir / f"FOOOF_boxplot_{metric}.svg",
                dpi=600,
                bbox_inches="tight",
                transparent=True,
            )
            plt.close(fig)
    METRIC_LABELS = {
        "exponent": "Exponent",
        "offset": "Offset",
        "alpha_pw": "Alpha-peak power",
    }
    METRICS = ["exponent", "offset", "alpha_pw"]
    COMPARISONS = [
        {
            "comparison": "NCBP_trans_vs_NCBP_nontrans",
            "first_group": "NCBP_trans",
            "second_group": "NCBP_nontrans",
            "label": "Transitioners − Non-transitioners",
        },
        {
            "comparison": "NCBP_trans_vs_HC",
            "first_group": "NCBP_trans",
            "second_group": "HC",
            "label": "Transitioners − HC",
        },
        {
            "comparison": "NCBP_nontrans_vs_HC",
            "first_group": "NCBP_nontrans",
            "second_group": "HC",
            "label": "Non-transitioners − HC",
        },
    ]
    fooof_schaefer_atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=100, yeo_networks=7, resolution_mm=2
    )
    fooof_schaefer_img = nib.load(fooof_schaefer_atlas.maps)
    fooof_schaefer_data = np.asarray(fooof_schaefer_img.get_fdata(), dtype=np.int16)

    def normalize_schaefer_label(label):
        if isinstance(label, bytes):
            return label.decode("utf-8")
        return str(label)

    fooof_schaefer_labels = [
        normalize_schaefer_label(label) for label in fooof_schaefer_atlas.labels
    ]
    max_atlas_value = int(np.nanmax(fooof_schaefer_data))
    if len(fooof_schaefer_labels) == max_atlas_value:
        atlas_values = list(range(1, max_atlas_value + 1))
    elif len(fooof_schaefer_labels) == max_atlas_value + 1:
        atlas_values = list(range(0, max_atlas_value + 1))
    else:
        raise ValueError(
            "Number of Schaefer labels does not match atlas integer values."
        )
    fooof_schaefer_label_to_value = dict(zip(fooof_schaefer_labels, atlas_values))
    data_roi_names = set(fooof_roi_plot["roi"].dropna().astype(str).unique())
    atlas_roi_names = set(fooof_schaefer_label_to_value)
    missing_roi_names = sorted(data_roi_names - atlas_roi_names)
    if missing_roi_names:
        raise ValueError(
            f"Some FOOOF ROI names do not match Schaefer atlas labels. First unmatched names: {missing_roi_names[:10]}"
        )

    def roi_values_to_schaefer_img(roi_value_df, value_col):
        output_data = np.zeros(fooof_schaefer_data.shape, dtype=np.float32)
        for row in roi_value_df[["roi", value_col]].itertuples(index=False):
            roi_name = str(row.roi)
            value = getattr(row, value_col)
            if roi_name not in fooof_schaefer_label_to_value:
                continue
            if not np.isfinite(value):
                continue
            atlas_value = fooof_schaefer_label_to_value[roi_name]
            if atlas_value == 0:
                continue
            output_data[fooof_schaefer_data == atlas_value] = float(value)
        return nib.Nifti1Image(
            output_data,
            affine=fooof_schaefer_img.affine,
            header=fooof_schaefer_img.header,
        )

    def safe_limits(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return (0.0, 1.0)
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if np.isclose(vmin, vmax):
            delta = max(abs(vmin), 1.0) * 0.05
            vmin -= delta
            vmax += delta
        return (vmin, vmax)

    for spec in COMPARISONS:
        first = spec["first_group"]
        second = spec["second_group"]
        comparison = spec["comparison"]
        metrics_here = [metric for metric in METRICS if metric in METRIC_LABELS]
        if not metrics_here:
            continue
        effects_comparison = fooof_roi_adjusted_effects_df.loc[
            fooof_roi_adjusted_effects_df["comparison"].eq(comparison)
        ].copy()
        effect_values = effects_comparison.loc[
            effects_comparison["metric"].isin(metrics_here), "adjusted_hedges_g"
        ].to_numpy(dtype=float)
        effect_values = effect_values[np.isfinite(effect_values)]
        if effect_values.size:
            effect_limit = float(np.nanmax(np.abs(effect_values)))
        else:
            effect_limit = 1.0
        if not np.isfinite(effect_limit) or effect_limit <= 0:
            effect_limit = 1.0
        effect_limit = max(effect_limit, 0.25)
        n_rows = len(metrics_here)
        fig_height = 3.0 * n_rows + 1.6
        fig = plt.figure(figsize=(15.5, fig_height), facecolor="white")
        fig.suptitle(
            f"{GROUP_LABELS[first]} vs {GROUP_LABELS[second]}",
            fontsize=22,
            fontweight="bold",
            y=0.98,
        )
        column_x = [0.055, 0.255, 0.555, 0.755]
        brain_w = 0.165
        top_margin = 0.88
        bottom_margin = 0.08
        available = top_margin - bottom_margin
        row_step = available / n_rows
        brain_h = min(0.23, row_step * 0.78)
        row_y = [
            top_margin - (i + 1) * row_step + (row_step - brain_h) / 2
            for i in range(n_rows)
        ]
        column_titles = [
            GROUP_LABELS[first],
            GROUP_LABELS[second],
            "Effect size",
            "Sig. effect size",
        ]
        for x, title in zip(column_x, column_titles):
            fig.text(
                x + brain_w / 2,
                0.90,
                title,
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="bold",
            )
        for row_i, metric in enumerate(metrics_here):
            y = row_y[row_i]
            first_df = fooof_roi_adjusted_means_df.loc[
                fooof_roi_adjusted_means_df["metric"].eq(metric)
                & fooof_roi_adjusted_means_df["analysis_group"].eq(first),
                ["roi", "adjusted_mean"],
            ].copy()
            second_df = fooof_roi_adjusted_means_df.loc[
                fooof_roi_adjusted_means_df["metric"].eq(metric)
                & fooof_roi_adjusted_means_df["analysis_group"].eq(second),
                ["roi", "adjusted_mean"],
            ].copy()
            mean_values = np.concatenate(
                [
                    first_df["adjusted_mean"].to_numpy(dtype=float),
                    second_df["adjusted_mean"].to_numpy(dtype=float),
                ]
            )
            mean_vmin, mean_vmax = safe_limits(mean_values)
            effect_df = fooof_roi_adjusted_effects_df.loc[
                fooof_roi_adjusted_effects_df["comparison"].eq(comparison)
                & fooof_roi_adjusted_effects_df["metric"].eq(metric),
                [
                    "roi",
                    "adjusted_hedges_g",
                    "p_fdr",
                    "significant_fdr",
                    "interaction_p_corrected",
                    "passes_interaction_gate",
                ],
            ].copy()
            sig_effect_df = effect_df[
                [
                    "roi",
                    "adjusted_hedges_g",
                    "significant_fdr",
                    "passes_interaction_gate",
                ]
            ].copy()
            sig_effect_df["sig_adjusted_hedges_g"] = np.where(
                sig_effect_df["significant_fdr"].eq(True)
                & sig_effect_df["passes_interaction_gate"].eq(True),
                sig_effect_df["adjusted_hedges_g"],
                0.0,
            )
            first_img = roi_values_to_schaefer_img(first_df, value_col="adjusted_mean")
            second_img = roi_values_to_schaefer_img(
                second_df, value_col="adjusted_mean"
            )
            effect_img = roi_values_to_schaefer_img(
                effect_df, value_col="adjusted_hedges_g"
            )
            sig_effect_img = roi_values_to_schaefer_img(
                sig_effect_df, value_col="sig_adjusted_hedges_g"
            )
            for x, img in zip(column_x[:2], [first_img, second_img]):
                plotting.plot_glass_brain(
                    img,
                    axes=[x, y, brain_w, brain_h],
                    display_mode="z",
                    cmap="turbo",
                    symmetric_cbar=False,
                    vmin=mean_vmin,
                    vmax=mean_vmax,
                    colorbar=False,
                    annotate=False,
                    black_bg=False,
                    plot_abs=False,
                    threshold=1e-12,
                    figure=fig,
                )
            plotting.plot_glass_brain(
                effect_img,
                axes=[column_x[2], y, brain_w, brain_h],
                display_mode="z",
                cmap="turbo",
                symmetric_cbar=True,
                vmin=-effect_limit,
                vmax=effect_limit,
                colorbar=False,
                annotate=False,
                black_bg=False,
                plot_abs=False,
                threshold=1e-12,
                figure=fig,
            )
            plotting.plot_glass_brain(
                sig_effect_img,
                axes=[column_x[3], y, brain_w, brain_h],
                display_mode="z",
                cmap="turbo",
                symmetric_cbar=True,
                vmin=-effect_limit,
                vmax=effect_limit,
                colorbar=False,
                annotate=False,
                black_bg=False,
                plot_abs=False,
                threshold=1e-12,
                figure=fig,
            )
            fig.text(
                0.03,
                y + brain_h / 2,
                METRIC_LABELS[metric],
                rotation=90,
                va="center",
                ha="center",
                fontsize=22,
                fontweight="bold",
            )
            mean_mappable = cm.ScalarMappable(
                norm=Normalize(vmin=mean_vmin, vmax=mean_vmax), cmap="turbo"
            )
            mean_mappable.set_array([])
            cax_mean = fig.add_axes([0.455, y + brain_h * 0.12, 0.014, brain_h * 0.76])
            cbar_mean = fig.colorbar(mean_mappable, cax=cax_mean)
            cbar_mean.ax.tick_params(labelsize=16)
            cbar_mean.set_label(METRIC_LABELS[metric], fontsize=20, labelpad=12)
        effect_mappable = cm.ScalarMappable(
            norm=TwoSlopeNorm(vmin=-effect_limit, vcenter=0.0, vmax=effect_limit),
            cmap="turbo",
        )
        effect_mappable.set_array([])
        cax_effect = fig.add_axes(
            [0.945, bottom_margin + 0.08, 0.015, available - 0.16]
        )
        cbar_effect = fig.colorbar(effect_mappable, cax=cax_effect)
        cbar_effect.ax.tick_params(labelsize=20)
        cbar_effect.set_label("Adjusted Hedges' g", fontsize=24, labelpad=12)
        fig.savefig(
            fooof_glass_dir / f"FOOOF_glass_{comparison}_means_effect_sig.svg",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
