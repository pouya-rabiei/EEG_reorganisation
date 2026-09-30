"""Whole-brain clustering and efficiency GEE."""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from config import comparison_dir, comparison_pairs, comparison_specs


def main():
    df_net_metric_groups = pd.read_csv(
        comparison_dir / "whole_brain" / "whole_brain_metrics.csv"
    )
    WHOLE_BRAIN_GROUPS = ["HC", "NCBP_nontrans", "NCBP_trans"]
    WHOLE_BRAIN_BANDS = ["Theta", "Alpha", "Beta", "Gamma"]
    comparison_pairs = [
        (spec["first_group"], spec["second_group"]) for spec in comparison_specs
    ]
    metrics = ["global_clustering", "global_efficiency"]
    whole_brain_dir = Path(comparison_dir) / "whole_brain"
    whole_brain_dir.mkdir(parents=True, exist_ok=True)

    def run_models_gee_whole_brain(data, metrics):
        term_results = []
        band_results = []
        for metric in metrics:
            df = data.loc[
                data["analysis_group"].isin(WHOLE_BRAIN_GROUPS)
                & data["freq_band"].isin(WHOLE_BRAIN_BANDS)
            ].copy()
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
            df[[metric, "age"]] = df[[metric, "age"]].replace([np.inf, -np.inf], np.nan)
            df = df.dropna(
                subset=[
                    "sample_id",
                    "analysis_group",
                    "freq_band",
                    "age",
                    "sex",
                    metric,
                ]
            ).copy()
            if df.empty:
                raise ValueError(f"No valid observations for {metric}")
            if df.duplicated(["sample_id", "freq_band"]).any():
                raise ValueError(f"Duplicate participant-band rows for {metric}")
            df["analysis_group"] = pd.Categorical(
                df["analysis_group"], categories=WHOLE_BRAIN_GROUPS
            )
            df["freq_band"] = pd.Categorical(
                df["freq_band"], categories=WHOLE_BRAIN_BANDS
            )
            cell_counts = df.groupby(
                ["analysis_group", "freq_band"], observed=False
            ).size()
            if cell_counts.eq(0).any():
                raise ValueError(
                    f"At least one group-band combination is empty for {metric}"
                )
            formula = f"{metric} ~ analysis_group * freq_band + age + C(sex)"
            result = smf.gee(
                formula=formula,
                groups="sample_id",
                data=df,
                family=sm.families.Gaussian(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit(cov_type="robust", maxiter=100)
            if not result.converged:
                raise RuntimeError(f"GEE did not converge for {metric}")
            wt_table = (
                result.wald_test_terms(scalar=True)
                .summary_frame()
                .reset_index()
                .rename(
                    columns={"index": "term", "P>chi2": "p_value", "pvalue": "p_value"}
                )
            )
            wt_table["metric"] = metric
            wt_table["model"] = "Three-group GEE"
            wt_table["formula"] = formula
            wt_table["n_rows"] = len(df)
            wt_table["n_subjects"] = df["sample_id"].nunique()
            wt_table["converged"] = bool(result.converged)
            term_results.append(wt_table)
            param_names = list(result.params.index)

            def group_vector(group_name, band):
                vector = np.zeros(len(param_names))
                if group_name != "HC":
                    main_term = f"analysis_group[T.{group_name}]"
                    vector[param_names.index(main_term)] = 1.0
                    if band != WHOLE_BRAIN_BANDS[0]:
                        interaction_term = (
                            f"analysis_group[T.{group_name}]:freq_band[T.{band}]"
                        )
                        vector[param_names.index(interaction_term)] = 1.0
                return vector

            for band in WHOLE_BRAIN_BANDS:
                band_desc = df.loc[df["freq_band"].eq(band)]
                for first, second in comparison_pairs:
                    L = group_vector(first, band) - group_vector(second, band)
                    test_res = result.t_test(L)
                    ci = np.asarray(test_res.conf_int()).reshape(-1)
                    first_values = band_desc.loc[
                        band_desc["analysis_group"].eq(first), metric
                    ]
                    second_values = band_desc.loc[
                        band_desc["analysis_group"].eq(second), metric
                    ]
                    band_results.append(
                        {
                            "comparison": f"{first} vs {second}",
                            "first_group": first,
                            "second_group": second,
                            "metric": metric,
                            "band": band,
                            "contrast": "First_minus_second_within_band",
                            "n_first": int(first_values.size),
                            "mean_first": float(first_values.mean()),
                            "sd_first": float(first_values.std(ddof=1)),
                            "n_second": int(second_values.size),
                            "mean_second": float(second_values.mean()),
                            "sd_second": float(second_values.std(ddof=1)),
                            "beta": float(np.squeeze(test_res.effect)),
                            "se": float(np.squeeze(test_res.sd)),
                            "z": float(np.squeeze(test_res.tvalue)),
                            "p_value": float(np.squeeze(test_res.pvalue)),
                            "ci_low": float(ci[0]),
                            "ci_high": float(ci[1]),
                            "n_rows_model": int(len(df)),
                            "n_subjects_model": int(df["sample_id"].nunique()),
                        }
                    )
        return (pd.concat(term_results, ignore_index=True), pd.DataFrame(band_results))

    def correct_term_results_by_family(df, correction="fdr_bh"):
        df = df.copy()
        df["p_corrected"] = np.nan
        for _, sub_df in df.groupby(["con_type", "term"], sort=False):
            valid = sub_df.loc[np.isfinite(sub_df["p_value"].astype(float))]
            if not valid.empty:
                df.loc[valid.index, "p_corrected"] = multipletests(
                    valid["p_value"].astype(float), method=correction
                )[1]
        df["significant_corrected"] = np.where(df["p_corrected"] < 0.05, "Yes", "No")
        return df

    def correct_band_results_by_metric_family(df, correction="fdr_bh"):
        df = df.copy()
        df["p_corrected"] = np.nan
        for _, sub_df in df.groupby(["comparison", "con_type", "metric"], sort=False):
            valid = sub_df.loc[np.isfinite(sub_df["p_value"].astype(float))]
            if not valid.empty:
                df.loc[valid.index, "p_corrected"] = multipletests(
                    valid["p_value"].astype(float), method=correction
                )[1]
        df["significant_corrected"] = np.where(df["p_corrected"] < 0.05, "Yes", "No")
        return df

    df_whole_brain = df_net_metric_groups.copy()
    all_terms = []
    all_bands = []
    for con in ["aec", "dwpli"]:
        df_con = df_whole_brain.loc[df_whole_brain["con_type"].eq(con)].copy()
        terms, bands = run_models_gee_whole_brain(data=df_con, metrics=metrics)
        terms["con_type"] = con
        bands["con_type"] = con
        all_terms.append(terms)
        all_bands.append(bands)
    terms_all = pd.concat(all_terms, ignore_index=True)
    bands_all = pd.concat(all_bands, ignore_index=True)
    terms_all = correct_term_results_by_family(terms_all, correction="fdr_bh")
    bands_all = correct_band_results_by_metric_family(bands_all, correction="fdr_bh")
    interaction_results = terms_all.loc[
        terms_all["term"].eq("analysis_group:freq_band"),
        ["con_type", "metric", "p_corrected"],
    ].rename(columns={"p_corrected": "interaction_p_corrected"})
    bands_all = bands_all.merge(
        interaction_results,
        on=["con_type", "metric"],
        how="left",
        validate="many_to_one",
    )
    # Both the interaction and the contrast must pass FDR correction.
    bands_all["passes_interaction_gate"] = bands_all["p_corrected"].lt(
        0.05
    ) & bands_all["interaction_p_corrected"].lt(0.05)
    terms_aec = terms_all.loc[terms_all["con_type"].eq("aec")].copy()
    terms_dwpli = terms_all.loc[terms_all["con_type"].eq("dwpli")].copy()
    band_aec = bands_all.loc[bands_all["con_type"].eq("aec")].copy()
    band_dwpli = bands_all.loc[bands_all["con_type"].eq("dwpli")].copy()
    terms_aec.to_csv(whole_brain_dir / "whole_brain_aec_GEE_omnibus.csv", index=False)
    band_aec.to_csv(
        whole_brain_dir / "whole_brain_aec_GEE_band_contrasts.csv", index=False
    )
    terms_dwpli.to_csv(
        whole_brain_dir / "whole_brain_dwpli_GEE_omnibus.csv", index=False
    )
    band_dwpli.to_csv(
        whole_brain_dir / "whole_brain_dwpli_GEE_band_contrasts.csv", index=False
    )


if __name__ == "__main__":
    main()
