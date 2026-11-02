"""Paired baseline and six-month pain rainclouds by transition group."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import basepath, figure_dir, load_subject_info, GROUP_LABELS, group_palette
from plotting_utils import save_plot


def plot_pain_raincloud(data, output_dir):
    groups = ["NCBP_trans", "NCBP_nontrans"]
    rng = np.random.default_rng(42)
    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, group in zip(axes, groups):
        paired = data.loc[data.analysis_group.eq(group), ["pain", "pain_6m"]]
        color = group_palette[group]
        jitter = rng.uniform(.15, .34, len(paired))
        for offset, values in zip(jitter, paired.to_numpy(float)):
            ax.plot([offset, 1 + offset], values, color=".65", alpha=.25,
                    linewidth=.6, zorder=1)
        for x, column in enumerate(["pain", "pain_6m"]):
            values = paired[column].to_numpy(float)
            if not len(values):
                continue
            if len(values) > 1 and np.ptp(values) > 0:
                violin = ax.violinplot(values, positions=[x], widths=.72,
                                      showmeans=False, showmedians=False,
                                      showextrema=False)
                body = violin["bodies"][0]
                vertices = body.get_paths()[0].vertices
                vertices[:, 0] = np.minimum(vertices[:, 0], x)
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(.8)
            ax.boxplot([values], positions=[x + .06], widths=.10,
                       patch_artist=True, manage_ticks=False, showfliers=False,
                       boxprops={"facecolor": color, "edgecolor": "black", "alpha": .8},
                       medianprops={"color": "black", "linewidth": 1.6},
                       whiskerprops={"color": "black"}, capprops={"color": "black"})
            ax.scatter(x + jitter, values, s=22, color=color, alpha=1,
                       edgecolors="none", zorder=4)
        ax.set_xticks([0, 1], ["Baseline", "6 months"])
        ax.set(xlim=(-.45, 1.45), ylim=(-.25, 10.25))
        ax.set_yticks(np.arange(0, 11, 2))
        ax.set_title(GROUP_LABELS[group], fontsize=20, pad=15)
        ax.tick_params(labelsize=16)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=".9")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pain intensity (0–10)", fontsize=18)
    fig.tight_layout(w_pad=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_plot(fig, output_dir / "pain_trajectory_raincloud")


def main():
    demo = pd.read_csv(basepath / "RAW/raw_data_ncbp/demo_ncbp.csv",
                       dtype={"subject": str})
    demo = demo.dropna(subset=["subject"]).copy()
    demo["subject"] = demo.subject.str.strip()
    demo = demo.loc[demo.subject.ne("")]
    info = load_subject_info()
    patients = info.loc[info.group.eq("ncbp"), ["subject", "analysis_group"]]
    data = patients.merge(demo[["subject", "pain", "pain_6m"]], on="subject",
                          how="left", validate="one_to_one")
    data[["pain", "pain_6m"]] = data[["pain", "pain_6m"]].apply(
        pd.to_numeric, errors="raise")
    if ((data[["pain", "pain_6m"]] < 0) | (data[["pain", "pain_6m"]] > 10)).any().any():
        raise ValueError("Pain scores must be on the 0–10 scale.")
    data = data.dropna(subset=["pain", "pain_6m"])
    if data.empty:
        raise ValueError("No eligible participants have pain scores at both visits.")
    plot_pain_raincloud(data, figure_dir / "pain_trajectory")


if __name__ == "__main__":
    main()
