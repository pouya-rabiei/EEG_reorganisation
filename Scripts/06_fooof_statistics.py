"""Network GEE and parcel follow-up spectral parameter statistics."""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from config import (
    ANALYSIS_GROUPS,
    PLOT_GROUPS,
    comparison_specs,
    network_names,
    spectral_dir,
)


def main():
    spectral_dir.mkdir(parents=True, exist_ok=True)
    fooof_net_all = pd.read_csv(spectral_dir / "fooof_network_summary.csv")
    fooof_roi_all = pd.read_csv(spectral_dir / "fooof_roi_all.csv")
    ANALYSIS_GROUPS = ["HC", "NCBP_nontrans", "NCBP_trans"]
    fooof_network_metrics = ["exponent_mean", "offset_mean", "alpha_pw_mean"]
    metric_display_names = {
        "exponent_mean": "Exponent",
        "offset_mean": "Offset",
        "alpha_pw_mean": "Mean alpha peak height",
    }

    def run_fooof_network_gee(data, metric_col, network_names, comparison_specs):
        df = data.copy()
        df = df[df["analysis_group"].isin(ANALYSIS_GROUPS)].copy()
        df = df[df["network"].isin(network_names)].copy()
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df = df.dropna(
            subset=["sample_id", "analysis_group", "network", "age", "sex", metric_col]
        ).copy()
        if df.empty:
            raise ValueError(f"No valid observations for {metric_col}")
        duplicate_keys = ["sample_id", "network"]
        if df.duplicated(duplicate_keys).any():
            duplicates = df.loc[
                df.duplicated(duplicate_keys, keep=False), duplicate_keys
            ]
            raise ValueError(
                f"Duplicate participant-network rows detected:\n{duplicates.head()}"
            )
        df["analysis_group"] = pd.Categorical(
            df["analysis_group"], categories=ANALYSIS_GROUPS, ordered=False
        )
        df["network"] = pd.Categorical(
            df["network"], categories=network_names, ordered=False
        )
        reference_network = network_names[0]
        df["age_c"] = df["age"] - df["age"].mean()
        formula = f"{metric_col} ~ analysis_group * network + age_c + C(sex)"
        result = smf.gee(
            formula=formula,
            groups="sample_id",
            data=df,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit(cov_type="robust")
        if not result.converged:
            raise RuntimeError(f"GEE did not converge: {metric_col}")
        model_info = pd.DataFrame(
            [
                {
                    "metric": metric_col,
                    "formula": formula,
                    "n_rows": int(df.shape[0]),
                    "n_subjects": int(df["sample_id"].nunique()),
                    "n_HC": int(
                        df.loc[df["analysis_group"].eq("HC"), "sample_id"].nunique()
                    ),
                    "n_NCBP_nontrans": int(
                        df.loc[
                            df["analysis_group"].eq("NCBP_nontrans"), "sample_id"
                        ].nunique()
                    ),
                    "n_NCBP_trans": int(
                        df.loc[
                            df["analysis_group"].eq("NCBP_trans"), "sample_id"
                        ].nunique()
                    ),
                    "covariance_structure": "Exchangeable",
                    "covariance_estimator": "Robust sandwich",
                }
            ]
        )
        wt = result.wald_test_terms(scalar=True)
        term_results = (
            wt.summary_frame().reset_index().rename(columns={"index": "term"})
        )
        if "P>chi2" in term_results.columns:
            term_results = term_results.rename(columns={"P>chi2": "p_value"})
        elif "pvalue" in term_results.columns:
            term_results = term_results.rename(columns={"pvalue": "p_value"})
        term_results["metric"] = metric_col
        term_results["n_rows"] = df.shape[0]
        term_results["n_subjects"] = df["sample_id"].nunique()
        param_names = list(result.params.index)
        contrast_rows = []

        def add_group_to_contrast(L, group_name, network, weight):
            if group_name == "HC":
                return
            main_term = f"analysis_group[T.{group_name}]"
            L[param_names.index(main_term)] += weight
            if network != reference_network:
                interaction_term = f"{main_term}:network[T.{network}]"
                L[param_names.index(interaction_term)] += weight

        for network in network_names:
            network_df = df.loc[df["network"].eq(network)].copy()
            for spec in comparison_specs:
                first = spec["first_group"]
                second = spec["second_group"]
                L = np.zeros(len(param_names))
                add_group_to_contrast(
                    L=L, group_name=first, network=network, weight=1.0
                )
                add_group_to_contrast(
                    L=L, group_name=second, network=network, weight=-1.0
                )
                test_res = result.t_test(L)
                ci = np.asarray(test_res.conf_int()).squeeze()
                first_values = (
                    network_df.loc[network_df["analysis_group"].eq(first), metric_col]
                    .dropna()
                    .to_numpy(dtype=float)
                )
                second_values = (
                    network_df.loc[network_df["analysis_group"].eq(second), metric_col]
                    .dropna()
                    .to_numpy(dtype=float)
                )
                contrast_rows.append(
                    {
                        "comparison": spec["comparison_display"],
                        "first_group": first,
                        "second_group": second,
                        "metric": metric_col,
                        "network": network,
                        "n_first": int(first_values.size),
                        "mean_first": (
                            float(np.mean(first_values))
                            if first_values.size
                            else np.nan
                        ),
                        "sd_first": (
                            float(np.std(first_values, ddof=1))
                            if first_values.size >= 2
                            else np.nan
                        ),
                        "n_second": int(second_values.size),
                        "mean_second": (
                            float(np.mean(second_values))
                            if second_values.size
                            else np.nan
                        ),
                        "sd_second": (
                            float(np.std(second_values, ddof=1))
                            if second_values.size >= 2
                            else np.nan
                        ),
                        "beta": float(np.squeeze(test_res.effect)),
                        "se": float(np.squeeze(test_res.sd)),
                        "z": float(np.squeeze(test_res.tvalue)),
                        "p_value": float(np.squeeze(test_res.pvalue)),
                        "ci95_low": float(ci[0]),
                        "ci95_high": float(ci[1]),
                        "n_rows_model": int(df.shape[0]),
                        "n_subjects_model": int(df["sample_id"].nunique()),
                    }
                )
        contrast_results = pd.DataFrame(contrast_rows)
        return (model_info, term_results, contrast_results)

    def correct_fooof_omnibus_by_family(df, correction="fdr_bh"):
        df = df.copy()
        df["p_corrected"] = np.nan
        family_cols = ["term"]
        for _, sub_df in df.groupby(family_cols, dropna=False, sort=False):
            valid = sub_df[sub_df["p_value"].notna()]
            if valid.empty:
                continue
            idx = valid.index
            df.loc[idx, "p_corrected"] = multipletests(
                valid["p_value"].astype(float), method=correction
            )[1]
        df["significant_corrected"] = np.where(df["p_corrected"] < 0.05, "Yes", "No")
        return df

    def correct_fooof_network_contrasts_by_family(df, correction="fdr_bh"):
        df = df.copy()
        df["p_corrected"] = np.nan
        family_cols = ["comparison", "metric"]
        for _, sub_df in df.groupby(family_cols, dropna=False, sort=False):
            valid = sub_df[sub_df["p_value"].notna()]
            if valid.empty:
                continue
            idx = valid.index
            df.loc[idx, "p_corrected"] = multipletests(
                valid["p_value"].astype(float), method=correction
            )[1]
        df["significant_corrected"] = np.where(df["p_corrected"] < 0.05, "Yes", "No")
        return df

    all_fooof_models = []
    all_fooof_terms = []
    all_fooof_contrasts = []
    for metric in fooof_network_metrics:
        model_df, terms_df, contrasts_df = run_fooof_network_gee(
            data=fooof_net_all,
            metric_col=metric,
            network_names=network_names,
            comparison_specs=comparison_specs,
        )
        all_fooof_models.append(model_df)
        all_fooof_terms.append(terms_df)
        all_fooof_contrasts.append(contrasts_df)
    fooof_models_df = pd.concat(all_fooof_models, ignore_index=True)
    fooof_terms_df = pd.concat(all_fooof_terms, ignore_index=True)
    fooof_contrasts_df = pd.concat(all_fooof_contrasts, ignore_index=True)
    fooof_models_df["metric_display"] = fooof_models_df["metric"].map(
        metric_display_names
    )
    fooof_terms_df["metric_display"] = fooof_terms_df["metric"].map(
        metric_display_names
    )
    fooof_contrasts_df["metric_display"] = fooof_contrasts_df["metric"].map(
        metric_display_names
    )
    fooof_terms_df = correct_fooof_omnibus_by_family(
        fooof_terms_df, correction="fdr_bh"
    )
    fooof_contrasts_df = correct_fooof_network_contrasts_by_family(
        fooof_contrasts_df, correction="fdr_bh"
    )
    fooof_interaction_results = fooof_terms_df.loc[
        fooof_terms_df["term"].eq("analysis_group:network"), ["metric", "p_corrected"]
    ].rename(columns={"p_corrected": "interaction_p_corrected"})
    fooof_contrasts_df = fooof_contrasts_df.merge(
        fooof_interaction_results, on="metric", how="left", validate="many_to_one"
    )
    # Both the interaction and the contrast must pass FDR correction.
    fooof_contrasts_df["passes_interaction_gate"] = fooof_contrasts_df[
        "p_corrected"
    ].lt(0.05) & fooof_contrasts_df["interaction_p_corrected"].lt(0.05)
    fooof_models_df.to_csv(
        spectral_dir / "FOOOF_network_GEE_model_info.csv", index=False
    )
    fooof_terms_df.to_csv(spectral_dir / "FOOOF_network_GEE_omnibus.csv", index=False)
    fooof_contrasts_df.to_csv(
        spectral_dir / "FOOOF_network_GEE_contrasts.csv", index=False
    )
    fooof_glass_dir = Path(spectral_dir) / "FOOOF_GLASS_BRAIN"
    fooof_glass_dir.mkdir(parents=True, exist_ok=True)
    METRIC_LABELS = {
        "exponent": "Exponent",
        "offset": "Offset",
        "alpha_pw": "Alpha-peak power",
    }
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
    fooof_roi_plot = fooof_roi_all.copy()
    fooof_roi_plot = fooof_roi_plot.loc[
        fooof_roi_plot["analysis_group"].isin(PLOT_GROUPS)
    ].copy()
    fooof_roi_plot["age"] = pd.to_numeric(fooof_roi_plot["age"], errors="coerce")
    for metric in METRIC_LABELS:
        fooof_roi_plot[metric] = pd.to_numeric(fooof_roi_plot[metric], errors="coerce")
    age_mean = fooof_roi_plot["age"].dropna().mean()
    fooof_roi_plot["age_c"] = fooof_roi_plot["age"] - age_mean
    fooof_roi_plot["analysis_group"] = pd.Categorical(
        fooof_roi_plot["analysis_group"],
        categories=["HC", "NCBP_nontrans", "NCBP_trans"],
        ordered=False,
    )

    def fit_fooof_roi_adjusted_model(data, roi_name, metric):
        df = data.loc[
            data["roi"].astype(str).eq(str(roi_name)),
            ["sample_id", "analysis_group", "age_c", "sex", metric],
        ].copy()
        df = df.dropna(subset=["sample_id", "analysis_group", "age_c", "sex", metric])
        if df.empty:
            return None
        counts = df.groupby("analysis_group", observed=True)["sample_id"].nunique()
        if not all(
            (group in counts.index and counts.loc[group] >= 3 for group in PLOT_GROUPS)
        ):
            return None
        formula = (
            f"{metric} ~ C(analysis_group, Treatment(reference='HC')) + age_c + C(sex)"
        )
        model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
        return (df, model)

    def adjusted_marginal_mean(model, model_df, group_name):
        new_df = model_df.copy()
        new_df["analysis_group"] = group_name
        predictions = model.predict(new_df)
        return float(np.nanmean(predictions))

    def get_adjusted_group_contrast(model, first_group, second_group):
        param_names = list(model.params.index)
        L = np.zeros(len(param_names), dtype=float)

        def add_group(group_name, weight):
            if group_name == "HC":
                return
            term = f"C(analysis_group, Treatment(reference='HC'))[T.{group_name}]"
            L[param_names.index(term)] += weight

        add_group(first_group, +1.0)
        add_group(second_group, -1.0)
        test = model.t_test(L)
        ci = np.asarray(test.conf_int()).squeeze()
        return {
            "beta": float(np.squeeze(test.effect)),
            "se": float(np.squeeze(test.sd)),
            "statistic": float(np.squeeze(test.tvalue)),
            "p_value": float(np.squeeze(test.pvalue)),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
        }

    def adjusted_hedges_g(model, beta):
        residual_sd = float(np.sqrt(model.mse_resid))
        if not np.isfinite(residual_sd) or residual_sd <= 0:
            return np.nan
        adjusted_d = beta / residual_sd
        df_resid = float(model.df_resid)
        if df_resid > 1:
            correction_j = 1.0 - 3.0 / (4.0 * df_resid - 1.0)
        else:
            correction_j = 1.0
        return float(correction_j * adjusted_d)

    adjusted_mean_rows = []
    adjusted_effect_rows = []
    roi_names = sorted(fooof_roi_plot["roi"].dropna().astype(str).unique())
    for metric in METRIC_LABELS:
        for roi_name in roi_names:
            fitted = fit_fooof_roi_adjusted_model(
                data=fooof_roi_plot, roi_name=roi_name, metric=metric
            )
            if fitted is None:
                continue
            model_df, model = fitted
            for group_name in PLOT_GROUPS:
                adjusted_mean = adjusted_marginal_mean(
                    model=model, model_df=model_df, group_name=group_name
                )
                n_group = int(
                    model_df.loc[
                        model_df["analysis_group"].eq(group_name), "sample_id"
                    ].nunique()
                )
                adjusted_mean_rows.append(
                    {
                        "metric": metric,
                        "roi": roi_name,
                        "analysis_group": group_name,
                        "adjusted_mean": adjusted_mean,
                        "n_group": n_group,
                        "n_total": int(model_df["sample_id"].nunique()),
                    }
                )
            for spec in COMPARISONS:
                first = spec["first_group"]
                second = spec["second_group"]
                contrast = get_adjusted_group_contrast(
                    model=model, first_group=first, second_group=second
                )
                g = adjusted_hedges_g(model=model, beta=contrast["beta"])
                adjusted_effect_rows.append(
                    {
                        "metric": metric,
                        "roi": roi_name,
                        "comparison": spec["comparison"],
                        "comparison_label": spec["label"],
                        "first_group": first,
                        "second_group": second,
                        "adjusted_beta": contrast["beta"],
                        "se": contrast["se"],
                        "statistic": contrast["statistic"],
                        "p_value": contrast["p_value"],
                        "ci95_low": contrast["ci95_low"],
                        "ci95_high": contrast["ci95_high"],
                        "adjusted_hedges_g": g,
                        "residual_sd": float(np.sqrt(model.mse_resid)),
                        "n_total": int(model_df["sample_id"].nunique()),
                    }
                )
    fooof_roi_adjusted_means_df = pd.DataFrame(adjusted_mean_rows)
    fooof_roi_adjusted_effects_df = pd.DataFrame(adjusted_effect_rows)
    fooof_roi_adjusted_effects_df["p_fdr"] = np.nan
    for (_, _), family in fooof_roi_adjusted_effects_df.groupby(
        ["comparison", "metric"], sort=False, observed=True
    ):
        valid = family["p_value"].notna()
        valid_index = family.index[valid]
        if len(valid_index) == 0:
            continue
        fooof_roi_adjusted_effects_df.loc[valid_index, "p_fdr"] = multipletests(
            fooof_roi_adjusted_effects_df.loc[valid_index, "p_value"].to_numpy(
                dtype=float
            ),
            method="fdr_bh",
        )[1]
    fooof_roi_adjusted_effects_df["significant_fdr"] = (
        fooof_roi_adjusted_effects_df["p_fdr"] < 0.05
    )
    fooof_network_plot_gates = fooof_contrasts_df[
        [
            "first_group",
            "second_group",
            "metric",
            "network",
            "interaction_p_corrected",
            "passes_interaction_gate",
        ]
    ].copy()
    fooof_network_plot_gates["metric"] = fooof_network_plot_gates[
        "metric"
    ].str.removesuffix("_mean")
    fooof_roi_adjusted_effects_df["network"] = (
        fooof_roi_adjusted_effects_df["roi"].astype(str).str.split("_").str[2]
    )
    fooof_roi_adjusted_effects_df = fooof_roi_adjusted_effects_df.merge(
        fooof_network_plot_gates,
        on=["first_group", "second_group", "metric", "network"],
        how="left",
        validate="many_to_one",
    )
    fooof_roi_adjusted_means_df.to_csv(
        fooof_glass_dir / "FOOOF_ROI_adjusted_means.csv", index=False
    )
    fooof_roi_adjusted_effects_df.to_csv(
        fooof_glass_dir / "FOOOF_ROI_adjusted_effect_sizes_FDR.csv", index=False
    )


if __name__ == "__main__":
    main()
