"""Within-network topology boxplots and brain maps."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns
import nibabel as nib
from nilearn import datasets, plotting
from config import (
    GROUP_LABELS,
    NETWORK_ORDER,
    comparison_dir,
    conn_labels,
    group_palette,
    network_names,
)


def main():
    df_within_net_topo = pd.read_csv(
        comparison_dir / "network_topology" / "subject_metrics.csv"
    )
    topo_contrasts_all = pd.read_csv(
        comparison_dir / "network_topology" / "GEE_contrasts.csv"
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
    GROUP_ORDER = ["NCBP_trans", "NCBP_nontrans", "HC"]
    group_palette = {
        "NCBP_trans": "#D55E00",
        "NCBP_nontrans": "#009E73",
        "HC": "#0072B2",
    }
    topology_metric_labels = {
        "within_network_clustering": "Clustering",
        "within_network_efficiency": "Efficiency",
    }
    network_figure_dir = Path(comparison_dir) / "network_topology" / "FIGURES"
    network_figure_dir.mkdir(parents=True, exist_ok=True)

    def plot_within_topology_box(df, con_type, metric, out_file):
        df_sub = df.loc[
            df["con_type"].eq(con_type)
            & df["analysis_group"].isin(GROUP_ORDER)
            & df["network"].isin(NETWORK_ORDER)
            & df["freq_band"].isin(BAND_ORDER)
        ].copy()
        for column in [metric, "age"]:
            df_sub[column] = pd.to_numeric(df_sub[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        df_sub = df_sub.dropna(
            subset=[metric, "age", "sex", "sample_id", "analysis_group"]
        )
        if df_sub.empty:
            return
        y_label = topology_metric_labels[metric]
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
        for ax, band in zip(axes.flat, BAND_ORDER):
            temp = df_sub.loc[df_sub["freq_band"].eq(band)]
            if temp.empty:
                ax.set_visible(False)
                continue
            sns.boxplot(
                data=temp,
                x="network",
                y=metric,
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
                linewidth=1.0,
                boxprops={"edgecolor": "black", "linewidth": 1.0, "alpha": 0.8},
                medianprops={"color": "black", "linewidth": 1.6},
                whiskerprops={"color": "black", "linewidth": 1.0},
                capprops={"color": "black", "linewidth": 1.0},
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=temp,
                x="network",
                y=metric,
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
                f"{conn_labels[con_type]} · {band} · {y_label}",
                fontsize=20,
                fontweight="bold",
                pad=12,
            )
            ax.set_ylabel(y_label, fontsize=18)
            ax.tick_params(axis="both", labelsize=16)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_axisbelow(True)
            ax.xaxis.grid(False)
            ax.yaxis.grid(True, color="0.85", linewidth=0.7)
            sns.despine(ax=ax)
        legend_handles = [
            Patch(facecolor=group_palette[group], label=GROUP_LABELS[group])
            for group in GROUP_ORDER
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=3,
            frameon=False,
            fontsize=20,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.89], h_pad=2.0, w_pad=2.0)
        fig.savefig(out_file, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(fig)

    for con_type in ["aec", "dwpli"]:
        for metric in topology_metric_labels:
            plot_within_topology_box(
                df=df_within_net_topo,
                con_type=con_type,
                metric=metric,
                out_file=network_figure_dir
                / f"network_three_groups_{con_type}_{metric}_boxplot.svg",
            )
    map_rows = topo_contrasts_all.loc[
        topo_contrasts_all["passes_interaction_gate"].eq(True)
        & topo_contrasts_all["metric"].isin(topology_metric_labels)
    ].copy()
    if map_rows.empty:
        pass
    else:
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=100, yeo_networks=7, resolution_mm=2, verbose=0
        )
        atlas_img = nib.load(atlas.maps)
        atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int16)
        atlas_lut = atlas.lut.loc[atlas.lut["index"].ne(0), ["index", "name"]].copy()
        atlas_lut["network"] = atlas_lut["name"].astype(str).str.split("_").str[2]
        colour_limit = float(map_rows["beta"].abs().max())
        colour_norm = TwoSlopeNorm(vmin=-colour_limit, vcenter=0, vmax=colour_limit)
        map_keys = ["first_group", "second_group", "con_type", "band", "metric"]
        for key, rows in map_rows.groupby(map_keys, sort=False, observed=True):
            first, second, con_type, band, metric = key
            effects = dict(zip(rows["network"], rows["beta"]))
            network_names = ", ".join(effects)
            effect_data = np.zeros(atlas_data.shape, dtype=np.float32)
            for parcel_value, network in atlas_lut[["index", "network"]].itertuples(
                index=False, name=None
            ):
                if network in effects:
                    effect_data[atlas_data == parcel_value] = effects[network]
            effect_img = nib.Nifti1Image(effect_data, atlas_img.affine)
            fig, ax = plt.subplots(figsize=(12, 5.5))
            fig.suptitle(
                f"{GROUP_LABELS[first]} vs {GROUP_LABELS[second]}",
                fontsize=20,
                fontweight="bold",
                y=0.98,
            )
            fig.text(
                0.5,
                0.88,
                f"{conn_labels[con_type]} · {band} · {topology_metric_labels[metric]} · "
                f"{network_names} {'network' if len(effects) == 1 else 'networks'}",
                ha="center",
                fontsize=18,
            )
            plotting.plot_glass_brain(
                effect_img,
                display_mode="lyrz",
                plot_abs=False,
                threshold=0,
                cmap="turbo",
                vmin=-colour_limit,
                vmax=colour_limit,
                colorbar=False,
                annotate=False,
                black_bg=False,
                axes=ax,
            )
            scalar_mappable = ScalarMappable(norm=colour_norm, cmap="turbo")
            scalar_mappable.set_array([])
            cax = fig.add_axes([0.2, 0.1, 0.6, 0.035])
            cbar = fig.colorbar(scalar_mappable, cax=cax, orientation="horizontal")
            cbar.set_label("Adjusted $\\beta$", fontsize=18)
            cbar.ax.tick_params(labelsize=16)
            fig.subplots_adjust(left=0.03, right=0.97, top=0.77, bottom=0.23)
            fig.savefig(
                network_figure_dir
                / f"network_{first}_vs_{second}_{con_type}_{band}_{metric}_brain.svg",
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
