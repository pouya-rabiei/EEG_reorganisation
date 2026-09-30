"""Within-network clustering and efficiency GEE."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from config import (
    ANALYSIS_GROUPS, NETWORK_ORDER, comparison_dir, comparison_specs,
    con_types, freq_bands,
)

def correct_fdr(df, family_cols, correction="fdr_bh"):
    df = df.copy()
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df["p_corrected"] = np.nan
    for _, family in df.groupby(family_cols, sort=False, observed=True):
        valid = family.loc[np.isfinite(family["p_value"])]
        if not valid.empty:
            df.loc[valid.index, "p_corrected"] = multipletests(
                valid["p_value"], method=correction
            )[1]
    df["significant_corrected"] = np.where(df["p_corrected"].lt(0.05), "Yes", "No")
    return df


def main():
    output_dir = comparison_dir / "network_topology"
    data = pd.read_csv(output_dir / "subject_metrics.csv")
    metrics = ["within_network_clustering", "within_network_efficiency"]
    analysis_type = "network_topology"
    data = data.loc[
        data["analysis_group"].isin(ANALYSIS_GROUPS)
        & data["freq_band"].isin(freq_bands)
        & data["network"].isin(NETWORK_ORDER)
    ].copy()
    data["analysis_group"] = pd.Categorical(
        data["analysis_group"], categories=ANALYSIS_GROUPS
    )
    data["network"] = pd.Categorical(data["network"], categories=NETWORK_ORDER)
    data["age"] = pd.to_numeric(data["age"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    model_rows = []
    term_tables = []
    contrast_rows = []
    for con_type in con_types:
        for metric in metrics:
            for band in freq_bands:
                df = data.loc[
                    data["con_type"].eq(con_type) & data["freq_band"].eq(band)
                ].copy()
                df[metric] = pd.to_numeric(df[metric], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                df = df.dropna(
                    subset=[
                        "sample_id",
                        "analysis_group",
                        "network",
                        "age",
                        "sex",
                        metric,
                    ]
                ).copy()
                model_label = f"{analysis_type} | {con_type} | {metric} | {band}"
                if df.empty:
                    raise ValueError(f"No valid observations: {model_label}")
                if df.duplicated(["sample_id", "network"]).any():
                    raise ValueError(
                        f"Duplicate participant-network rows: {model_label}"
                    )
                cell_counts = df.groupby(
                    ["analysis_group", "network"], observed=False
                ).size()
                if cell_counts.eq(0).any():
                    raise ValueError(
                        f"An expected group-network cell is empty: {model_label}"
                    )
                formula = f"{metric} ~ analysis_group * network + age + C(sex)"
                result = smf.gee(
                    formula=formula,
                    groups="sample_id",
                    data=df,
                    family=sm.families.Gaussian(),
                    cov_struct=sm.cov_struct.Exchangeable(),
                ).fit(cov_type="robust", maxiter=100)
                if not result.converged:
                    raise RuntimeError(f"GEE did not converge: {model_label}")
                model_metadata = {
                    "analysis_type": analysis_type,
                    "con_type": con_type,
                    "metric": metric,
                    "band": band,
                    "model": "Three-group GEE",
                    "n_rows": int(len(df)),
                    "n_subjects": int(df["sample_id"].nunique()),
                    "converged": bool(result.converged),
                }
                model_rows.append(
                    {
                        **model_metadata,
                        "formula": formula,
                        "reference_group": "HC",
                        "reference_location": NETWORK_ORDER[0],
                        **{
                            f"n_{group}": int(
                                df.loc[
                                    df["analysis_group"].eq(group), "sample_id"
                                ].nunique()
                            )
                            for group in ANALYSIS_GROUPS
                        },
                    }
                )
                terms = (
                    result.wald_test_terms(scalar=True)
                    .summary_frame()
                    .reset_index()
                    .rename(
                        columns={
                            "index": "term",
                            "P>chi2": "p_value",
                            "pvalue": "p_value",
                        }
                    )
                )
                for column, value in model_metadata.items():
                    terms[column] = value
                term_tables.append(terms)
                param_names = list(result.params.index)

                def group_vector(group, network):
                    vector = np.zeros(len(param_names))
                    if group != "HC":
                        main_term = f"analysis_group[T.{group}]"
                        vector[param_names.index(main_term)] = 1.0
                        if network != NETWORK_ORDER[0]:
                            interaction_term = (
                                f"{main_term}:network[T.{network}]"
                            )
                            vector[param_names.index(interaction_term)] = 1.0
                    return vector

                for network in NETWORK_ORDER:
                    network_df = df.loc[df["network"].eq(network)]
                    for spec in comparison_specs:
                        first = spec["first_group"]
                        second = spec["second_group"]
                        L = group_vector(first, network) - group_vector(
                            second, network
                        )
                        test_res = result.t_test(L)
                        ci = np.asarray(test_res.conf_int()).reshape(-1)
                        first_values = network_df.loc[
                            network_df["analysis_group"].eq(first), metric
                        ]
                        second_values = network_df.loc[
                            network_df["analysis_group"].eq(second), metric
                        ]
                        contrast_rows.append(
                            {
                                **model_metadata,
                                "comparison": spec["comparison_display"],
                                "first_group": first,
                                "second_group": second,
                                "network": network,
                                "contrast": "First_minus_second",
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
                            }
                        )
    models = pd.DataFrame(model_rows)
    terms = pd.concat(term_tables, ignore_index=True)
    contrasts = pd.DataFrame(contrast_rows)
    terms = correct_fdr(
        terms,
        family_cols=["analysis_type", "con_type", "term", "metric"],
        correction="fdr_bh",
    )
    contrasts = correct_fdr(
        contrasts,
        family_cols=["comparison", "analysis_type", "con_type", "metric", "band"],
        correction="fdr_bh",
    )
    model_keys = ["analysis_type", "con_type", "metric", "band"]
    interaction_results = terms.loc[
        terms["term"].eq("analysis_group:network"),
        model_keys + ["p_corrected"],
    ].rename(columns={"p_corrected": "interaction_p_corrected"})
    contrasts = contrasts.merge(
        interaction_results, on=model_keys, how="left", validate="many_to_one"
    )
    # Both the interaction and the contrast must pass FDR correction.
    contrasts["passes_interaction_gate"] = contrasts["p_corrected"].lt(
        0.05
    ) & contrasts["interaction_p_corrected"].lt(0.05)
    models.to_csv(output_dir / "model_info.csv", index=False)
    terms.to_csv(output_dir / "GEE_omnibus.csv", index=False)
    contrasts.to_csv(output_dir / "GEE_contrasts.csv", index=False)


if __name__ == "__main__":
    main()
