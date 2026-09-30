"""Three-group preprocessing quality control (QC) statistics."""

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from config import PREPROCESS_DIR, comparison_dir, load_subject_info


def main():
    QC_GROUPS = ["NCBP_trans", "NCBP_nontrans", "HC"]
    QC_PAIRS = [
        ("NCBP_trans", "NCBP_nontrans"),
        ("NCBP_trans", "HC"),
        ("NCBP_nontrans", "HC"),
    ]
    QC_METRICS = {
        "n_bad_chans": "Bad channels",
        "n_removed_icas_total": "Removed ICA components",
        "n_epochs_autoreject_dropped": "AutoReject-rejected epochs",
        "n_epochs_after_autoreject": "Retained epochs",
    }

    def run_group_qc_statistics(data, output_dir):
        descriptives, omnibus, pairwise = ([], [], [])
        for metric, label in QC_METRICS.items():
            values = {
                group: data.loc[data["analysis_group"].eq(group), metric].to_numpy()
                for group in QC_GROUPS
            }
            for group, x in values.items():
                descriptives.append(
                    {
                        "metric": metric,
                        "label": label,
                        "analysis_group": group,
                        "n": len(x),
                        "median": np.median(x),
                        "q1": np.quantile(x, 0.25),
                        "q3": np.quantile(x, 0.75),
                        "mean": np.mean(x),
                        "sd": np.std(x, ddof=1),
                        "min": np.min(x),
                        "max": np.max(x),
                    }
                )
            if np.ptp(np.concatenate(list(values.values()))) == 0:
                h_stat, p_value = (0.0, 1.0)
            else:
                h_stat, p_value = kruskal(*values.values())
            omnibus.append(
                {
                    "metric": metric,
                    "label": label,
                    "H": h_stat,
                    "df": 2,
                    "n_subjects": len(data),
                    "p_value": p_value,
                }
            )
            for first, second in QC_PAIRS:
                x, y = (values[first], values[second])
                u_stat, p_value = mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided",
                    method="asymptotic",
                    use_continuity=True,
                )
                pairwise.append(
                    {
                        "metric": metric,
                        "label": label,
                        "comparison": f"{first} vs {second}",
                        "first_group": first,
                        "second_group": second,
                        "n_first": len(x),
                        "n_second": len(y),
                        "median_first": np.median(x),
                        "median_second": np.median(y),
                        "q1_first": np.quantile(x, 0.25),
                        "q3_first": np.quantile(x, 0.75),
                        "q1_second": np.quantile(y, 0.25),
                        "q3_second": np.quantile(y, 0.75),
                        "U": u_stat,
                        "p_value": p_value,
                        "rank_biserial": 2 * u_stat / (len(x) * len(y)) - 1,
                    }
                )
        descriptives = pd.DataFrame(descriptives)
        omnibus = pd.DataFrame(omnibus)
        pairwise = pd.DataFrame(pairwise)
        omnibus["p_corrected"] = multipletests(omnibus["p_value"], method="fdr_bh")[1]
        pairwise["p_corrected"] = multipletests(pairwise["p_value"], method="fdr_bh")[1]
        omnibus["sig_corrected"] = omnibus["p_corrected"].lt(0.05)
        pairwise["significant_corrected"] = pairwise["p_corrected"].lt(0.05)
        pairwise["omnibus_p_corrected"] = pairwise["metric"].map(
            omnibus.set_index("metric")["p_corrected"]
        )
        pairwise["passes_omnibus_gate"] = pairwise["p_corrected"].lt(0.05) & pairwise[
            "omnibus_p_corrected"
        ].lt(0.05)
        descriptives.to_csv(
            output_dir / "qc_descriptives_three_groups.csv", index=False
        )
        omnibus.to_csv(output_dir / "qc_omnibus_three_groups.csv", index=False)
        pairwise.to_csv(output_dir / "qc_pairwise_three_groups.csv", index=False)
        return (descriptives, omnibus, pairwise)

    subject_info = load_subject_info()
    tables = []
    for group in ["ncbp", "ncbphc"]:
        table = pd.read_csv(
            PREPROCESS_DIR / f"preprocess_{group}" / f"process_stats_{group}.csv",
            dtype={"subject": str},
        )
        table["subject"] = table["subject"].str.strip()
        tables.append(table.assign(group=group))
    data = subject_info.merge(
        pd.concat(tables), on=["group", "subject"], validate="one_to_one"
    )
    output_dir = comparison_dir / "preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "preprocessing_subjects.csv", index=False)
    run_group_qc_statistics(data, output_dir)


if __name__ == "__main__":
    main()
