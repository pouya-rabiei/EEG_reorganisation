"""Age- and sex-adjusted PSD cluster statistics."""

import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from config import (
    ANALYSIS_GROUPS,
    N_RESAMPLES,
    RANGE,
    SEED,
    comparison_specs,
    network_names,
    spectral_dir,
)


def main():
    spectral_dir.mkdir(parents=True, exist_ok=True)
    psd_all = pd.read_csv(spectral_dir / "psd_interp_all_groups.csv")
    ANALYSIS_GROUPS = ["HC", "NCBP_nontrans", "NCBP_trans"]

    def build_network_spectrum_matrix(
        data_long, network_name, value_col, freq_min=RANGE[0], freq_max=RANGE[1]
    ):
        selector = "roi"
        meta_cols = ["sample_id", "analysis_group", "age", "sex"]
        frequencies = pd.to_numeric(data_long["frequency"]).round(4)
        selected = (
            data_long["roi"].astype(str).str.contains(f"_{network_name}_", regex=False)
        )
        selected &= frequencies.between(freq_min, freq_max)
        df = data_long.loc[
            selected, meta_cols + ["frequency", selector, value_col]
        ].copy()
        df["frequency"] = frequencies.loc[selected].to_numpy()
        df[value_col] = pd.to_numeric(df[value_col])
        if df.empty:
            raise ValueError(f"No spectral data for {network_name}")
        meta = (
            df[meta_cols]
            .drop_duplicates()
            .set_index("sample_id", verify_integrity=True)
        )
        n_before = len(meta)
        meta["age"] = pd.to_numeric(meta["age"])
        meta = meta.dropna(subset=["analysis_group", "age", "sex"])
        if not meta["analysis_group"].isin(ANALYSIS_GROUPS).all():
            raise ValueError("Unexpected analysis_group label")
        df = df.loc[df["sample_id"].isin(meta.index)]
        if df.duplicated(["sample_id", selector, "frequency"]).any():
            raise ValueError(f"Duplicate spectral rows in {network_name}")
        if not np.isfinite(df[value_col]).all():
            raise ValueError(f"Nonfinite spectral values in {network_name}")
        if (df[value_col] <= 0).any():
            raise ValueError("PSD must be positive")
        counts = df.groupby(["sample_id", "frequency"])["roi"].nunique()
        if not counts.eq(df["roi"].nunique()).all():
            raise ValueError(f"Incomplete ROI coverage in {network_name}")
        df_sub = (
            df.groupby(["sample_id", "frequency"], observed=True)[value_col]
            .mean()
            .reset_index()
        )
        df_sub[value_col] = np.log10(df_sub[value_col])
        wide = (
            df_sub.pivot(index="sample_id", columns="frequency", values=value_col)
            .sort_index()
            .sort_index(axis=1)
        )
        if wide.empty or wide.isna().any().any() or wide.shape[1] < 3:
            raise ValueError(f"Empty/incomplete frequency grid in {network_name}")
        return (
            wide.to_numpy(float),
            wide.columns.to_numpy(float),
            meta.loc[wide.index],
            n_before - len(wide),
        )

    def frequency_clusters(tvals, freqs, threshold):
        clusters = []
        step = np.min(np.diff(freqs))
        for sign in (1, -1):
            idx = np.flatnonzero(sign * tvals > threshold)
            if idx.size:
                cuts = (
                    np.flatnonzero(
                        (np.diff(idx) > 1) | (np.diff(freqs[idx]) > 1.5 * step)
                    )
                    + 1
                )
                clusters.extend(np.split(idx, cuts))
        masses = np.array([np.abs(tvals[idx]).sum() for idx in clusters])
        return (clusters, masses)

    def run_adjusted_spectrum_cluster_test(
        Y, freqs, meta, first, second, n_bootstrap=N_RESAMPLES, seed=SEED
    ):
        group = pd.get_dummies(meta["analysis_group"], dtype=float)
        if not set(ANALYSIS_GROUPS).issubset(group.columns):
            raise ValueError("All three analysis groups must be present")
        group = group[ANALYSIS_GROUPS[1:]].to_numpy()
        sex = pd.get_dummies(meta["sex"], drop_first=True, dtype=float).to_numpy()
        age = meta["age"].to_numpy(float)
        X = np.column_stack([np.ones(len(meta)), group, age - age.mean(), sex])
        n, p = X.shape
        if n <= p or np.linalg.matrix_rank(X) < p:
            raise ValueError("Insufficient data or confounded model predictors")
        c = np.zeros(p)
        if first != "HC":
            c[ANALYSIS_GROUPS.index(first)] += 1
        if second != "HC":
            c[ANALYSIS_GROUPS.index(second)] -= 1
        A = np.linalg.pinv(X)
        h = np.sum(X * A.T, axis=1)
        if np.any(h >= 1 - 1e-10):
            raise ValueError("Unit-leverage observation: HC3 is not estimable")
        weights = c @ A

        def fit_stat(values):
            coef = A @ values
            residual = values - X @ coef
            beta = c @ coef
            se = np.sqrt(
                np.sum((weights[:, None] * residual / (1 - h[:, None])) ** 2, axis=0)
            )
            if not np.isfinite(se).all() or np.any(se <= 0):
                raise ValueError("Non-estimable standard error")
            return (beta, beta / se)

        beta, t_obs = fit_stat(Y)
        threshold = float(t.ppf(0.975, n - p))
        clusters, masses = frequency_clusters(t_obs, freqs, threshold)
        v = A @ A.T @ c
        denominator = float(c @ v)
        coef = A @ Y
        coef_null = coef - np.outer(v, beta) / denominator
        fitted_null = X @ coef_null
        h_null = h - (X @ v) ** 2 / denominator
        residual_null = (Y - fitted_null) / (1 - h_null[:, None])
        rng = np.random.default_rng(seed)
        null_max = np.empty(n_bootstrap if clusters else 0)
        for b in range(len(null_max)):
            multipliers = rng.choice([-1.0, 1.0], size=(n, 1))
            Y_boot = fitted_null + residual_null * multipliers
            _, t_boot = fit_stat(Y_boot)
            _, boot_masses = frequency_clusters(t_boot, freqs, threshold)
            null_max[b] = boot_masses.max() if boot_masses.size else 0.0
        pvals = np.array(
            [
                (1 + np.count_nonzero(null_max >= mass)) / (n_bootstrap + 1)
                for mass in masses
            ]
        )
        return {
            "freqs": freqs,
            "T_obs": t_obs,
            "beta": beta,
            "clusters": [(idx,) for idx in clusters],
            "cluster_p_values": pvals,
            "cluster_masses": masses,
            "threshold": threshold,
            "n_first": int(meta["analysis_group"].eq(first).sum()),
            "n_second": int(meta["analysis_group"].eq(second).sum()),
            "n_total": n,
            "null_max": null_max,
        }

    def collect_network_spectrum_cluster_results(
        data_long,
        comparison_specs,
        network_names,
        value_col,
        analysis_name,
        freq_min=RANGE[0],
        freq_max=RANGE[1],
        n_bootstrap=N_RESAMPLES,
        correction="fdr_bh",
    ):
        raw_results = {spec["comparison_display"]: {} for spec in comparison_specs}
        summary_rows = []
        cluster_rows = []
        for network in network_names:
            Y, freqs, meta, n_excluded = build_network_spectrum_matrix(
                data_long, network, value_col, freq_min, freq_max
            )
            for spec in comparison_specs:
                comparison = spec["comparison_display"]
                res = run_adjusted_spectrum_cluster_test(
                    Y,
                    freqs,
                    meta,
                    first=spec["first_group"],
                    second=spec["second_group"],
                    n_bootstrap=n_bootstrap,
                )
                raw_results[comparison][network] = res
                pvals = res["cluster_p_values"]
                base = {
                    "analysis": analysis_name,
                    "comparison": comparison,
                    "network": network,
                    "method": "OLS_HC3_wild_bootstrap",
                }
                summary_rows.append(
                    {
                        **base,
                        "n_first": res["n_first"],
                        "n_second": res["n_second"],
                        "n_total": res["n_total"],
                        "n_excluded_covariates": n_excluded,
                        "freq_min": float(freqs.min()),
                        "freq_max": float(freqs.max()),
                        "threshold": res["threshold"],
                        "n_bootstrap": n_bootstrap,
                        "n_clusters": len(pvals),
                        "min_cluster_p": float(pvals.min()) if len(pvals) else 1.0,
                    }
                )
                for i, (idx,) in enumerate(res["clusters"]):
                    t_cluster = res["T_obs"][idx]
                    cluster_rows.append(
                        {
                            **base,
                            "cluster_index": i,
                            "p_cluster": float(pvals[i]),
                            "freq_min": float(freqs[idx].min()),
                            "freq_max": float(freqs[idx].max()),
                            "n_freqs": len(idx),
                            "cluster_mass": float(res["cluster_masses"][i]),
                            "t_min": float(t_cluster.min()),
                            "t_max": float(t_cluster.max()),
                            "t_mean": float(t_cluster.mean()),
                            "beta_mean": float(res["beta"][idx].mean()),
                        }
                    )
        summary = pd.DataFrame(summary_rows)
        summary["p_corrected"] = multipletests(
            summary["min_cluster_p"], method=correction
        )[1]
        summary["significant_corrected"] = summary["p_corrected"] < 0.05
        clusters = pd.DataFrame(cluster_rows)
        if clusters.empty:
            clusters = pd.DataFrame(
                columns=[
                    "analysis",
                    "comparison",
                    "network",
                    "method",
                    "cluster_index",
                    "p_cluster",
                    "freq_min",
                    "freq_max",
                    "n_freqs",
                    "cluster_mass",
                    "t_min",
                    "t_max",
                    "t_mean",
                    "beta_mean",
                ]
            )
        clusters = clusters.merge(
            summary[["comparison", "network", "p_corrected"]].rename(
                columns={"p_corrected": "network_p_fdr"}
            ),
            on=["comparison", "network"],
            how="left",
            validate="many_to_one",
        )
        return (raw_results, summary, clusters)

    raw_psd_results, psd_cluster_summary, psd_cluster_all = (
        collect_network_spectrum_cluster_results(
            data_long=psd_all,
            comparison_specs=comparison_specs,
            network_names=network_names,
            value_col="psd",
            analysis_name="interp_psd_age_sex_adjusted",
            freq_min=RANGE[0],
            freq_max=RANGE[1],
            n_bootstrap=N_RESAMPLES,
            correction="fdr_bh",
        )
    )
    psd_cluster_summary.to_csv(
        spectral_dir / "PSD_network_cluster_summary.csv", index=False
    )
    psd_cluster_all.to_csv(
        spectral_dir / "PSD_network_cluster_all.csv", index=False
    )


if __name__ == "__main__":
    main()
