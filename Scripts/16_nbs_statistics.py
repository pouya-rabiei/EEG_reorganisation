"""Unadjusted pooled-t NBS across three thresholds."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits
from config import (
    NBS_ALPHA,
    NBS_BANDS,
    NBS_BATCH_SIZE,
    NBS_COMPARISONS,
    NBS_CON_TYPES,
    NBS_K,
    NBS_SEED,
    NBS_THRESH_LIST,
    comparison_dir,
    derivpath,
    load_parcels,
    spectral_dir,
)


def main():

    def load_nbs_edges(metadata, source_paths, parcel_names, con, band):
        if metadata.duplicated(["group", "subject"]).any():
            raise ValueError("Duplicate source-group/subject keys in NBS metadata")
        edge_i, edge_j = np.triu_indices(len(parcel_names), k=1)
        vectors, participants, missing = ([], [], [])
        for record in metadata.to_dict("records"):
            source, subject = (record["group"], record["subject"])
            path = (
                Path(source_paths[source])
                / subject
                / f"{subject}_{con}_matrix_{band}.csv"
            )
            if not path.is_file():
                missing.append(
                    {
                        **record,
                        "con_type": con,
                        "freq_band": band,
                        "reason": "matrix file missing",
                        "path": str(path),
                    }
                )
                continue
            matrix = (
                pd.read_csv(path, index_col=0)
                .loc[parcel_names, parcel_names]
                .to_numpy(dtype=float)
            )
            if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
                raise ValueError(f"Expected a finite, symmetric matrix: {path}")
            vectors.append(matrix[edge_i, edge_j])
            participants.append(record)
        if not vectors:
            raise ValueError(f"No NBS input matrices for {con}, {band}")
        return (np.stack(vectors), pd.DataFrame(participants), missing)

    def nbs_pooled_t_from_membership(values, membership, n_first):
        n_total = values.shape[0]
        n_second = n_total - n_first
        centred = values - values.mean(axis=0)
        total_sum = centred.sum(axis=0)
        total_squares = (centred * centred).sum(axis=0)
        first_sum = membership @ centred
        first_squares = membership @ (centred * centred)
        second_sum = total_sum - first_sum
        second_squares = total_squares - first_squares
        variance = (
            first_squares
            - first_sum**2 / n_first
            + second_squares
            - second_sum**2 / n_second
        ) / (n_total - 2)
        denominator = np.sqrt(np.maximum(variance, 0) * (1 / n_first + 1 / n_second))
        difference = first_sum / n_first - second_sum / n_second
        return np.divide(
            difference,
            denominator,
            out=np.zeros_like(difference),
            where=denominator > 0,
        )

    def nbs_edge_components(t_values, threshold, edge_i, edge_j, n_rois, labels=True):
        active = np.flatnonzero(np.abs(t_values) > threshold)
        edge_labels = np.zeros(len(t_values), dtype=np.int32) if labels else None
        if active.size == 0:
            return (edge_labels, np.array([], dtype=int)) if labels else 0
        i, j = (edge_i[active], edge_j[active])
        graph = csr_matrix(
            (np.ones(2 * len(active)), (np.r_[i, j], np.r_[j, i])),
            shape=(n_rois, n_rois),
        )
        _, node_labels = connected_components(graph, directed=False)
        _, component_index = np.unique(node_labels[i], return_inverse=True)
        sizes = np.bincount(component_index)
        if not labels:
            return int(sizes.max())
        edge_labels[active] = component_index + 1
        return (edge_labels, sizes)

    def run_nbs_thresholds(
        first_values,
        second_values,
        n_rois,
        thresholds=NBS_THRESH_LIST,
        k=NBS_K,
        seed=NBS_SEED,
    ):
        n_first, n_second = (len(first_values), len(second_values))
        if min(n_first, n_second) < 2:
            raise ValueError("Each NBS group needs at least two participants")
        values = np.concatenate([first_values, second_values], axis=0)
        n_total = len(values)
        edge_i, edge_j = np.triu_indices(n_rois, k=1)
        observed_membership = np.zeros((1, n_total))
        observed_membership[0, :n_first] = 1
        observed_t = nbs_pooled_t_from_membership(values, observed_membership, n_first)[
            0
        ]
        observed = [
            nbs_edge_components(observed_t, t, edge_i, edge_j, n_rois)
            for t in thresholds
        ]
        null = np.zeros((len(thresholds), k), dtype=np.int32)
        rng = np.random.RandomState(seed)
        with threadpool_limits(limits=1):
            for start in range(0, k, NBS_BATCH_SIZE):
                count = min(NBS_BATCH_SIZE, k - start)
                membership = np.zeros((count, n_total))
                for index in range(count):
                    membership[index, rng.permutation(n_total)[:n_first]] = 1
                permuted_t = nbs_pooled_t_from_membership(values, membership, n_first)
                for index, t_values in enumerate(permuted_t):
                    for threshold_index, threshold in enumerate(thresholds):
                        null[threshold_index, start + index] = nbs_edge_components(
                            t_values, threshold, edge_i, edge_j, n_rois, labels=False
                        )
        results = []
        for index, (edge_labels, sizes) in enumerate(observed):
            pvalues = np.array(
                [
                    (1 + np.count_nonzero(null[index] >= size)) / (k + 1)
                    for size in sizes
                ]
            )
            results.append((edge_labels, sizes, pvalues, null[index]))
        return (observed_t, results)

    def run_three_group_nbs(metadata, source_paths, parcel_info, output_dir):
        output_dir = Path(output_dir)
        nbs_dir = output_dir / "nbs"
        nbs_dir.mkdir(parents=True, exist_ok=True)
        parcel_names = parcel_info["ROI Name"].astype(str).tolist()
        n_rois = len(parcel_names)
        edge_i, edge_j = np.triu_indices(n_rois, k=1)
        metadata = metadata.loc[
            metadata["analysis_group"].isin(["NCBP_trans", "NCBP_nontrans", "HC"])
        ].copy()
        metadata["subject"] = metadata["subject"].astype(str).str.strip()
        settings = {
            "thresholds": NBS_THRESH_LIST,
            "permutations": NBS_K,
            "seed": NBS_SEED,
            "comparisons": NBS_COMPARISONS,
            "statistic": "unpaired pooled-variance t",
            "covariates": [],
            "tail": "both (absolute t; mixed-sign components allowed)",
            "component_measure": "edge count",
            "component_alpha": NBS_ALPHA,
            "pvalue_formula": "(1 + null_max >= observed_size count) / (K + 1)",
            "correction_scope": "components within each comparison x measure x band x threshold",
            "thresholds_share_permutations": True,
        }
        with open(nbs_dir / "nbs_settings.json", "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2)
        all_components, all_edges, all_participants, all_missing = ([], [], [], [])
        for con in NBS_CON_TYPES:
            for band in NBS_BANDS:
                values, participants, missing = load_nbs_edges(
                    metadata, source_paths, parcel_names, con, band
                )
                all_missing.extend(missing)
                all_participants.append(
                    participants.assign(con_type=con, freq_band=band)
                )
                for first, second in NBS_COMPARISONS:
                    first_values = values[
                        participants["analysis_group"].eq(first).to_numpy()
                    ]
                    second_values = values[
                        participants["analysis_group"].eq(second).to_numpy()
                    ]
                    comparison = f"{first} vs {second}"
                    observed_t, results = run_nbs_thresholds(
                        first_values, second_values, n_rois
                    )
                    t_matrix = np.zeros((n_rois, n_rois))
                    t_matrix[edge_i, edge_j] = observed_t
                    t_matrix += t_matrix.T
                    for threshold, (edge_labels, sizes, pvalues, null) in zip(
                        NBS_THRESH_LIST, results
                    ):
                        threshold_text = str(float(threshold))
                        test_dir = nbs_dir / comparison / f"nbs_t_{threshold_text}"
                        test_dir.mkdir(parents=True, exist_ok=True)
                        stem = f"{comparison}_{con}_{band}_t{threshold_text}"
                        common = {
                            "comparison": comparison,
                            "first_group": first,
                            "second_group": second,
                            "con_type": con,
                            "freq_band": band,
                            "primary_threshold_t": threshold,
                            "n_first": len(first_values),
                            "n_second": len(second_values),
                            "n_permutations": NBS_K,
                            "seed": NBS_SEED,
                            "tail": "both",
                            "statistic": "pooled t (unadjusted)",
                        }
                        component_rows = []
                        for component_id, (size, pvalue) in enumerate(
                            zip(sizes, pvalues), start=1
                        ):
                            selected = edge_labels == component_id
                            component_t = observed_t[selected]
                            direction = (
                                "First < second"
                                if np.all(component_t < 0)
                                else (
                                    "First > second"
                                    if np.all(component_t > 0)
                                    else "Mixed"
                                )
                            )
                            component_rows.append(
                                {
                                    **common,
                                    "component_id": component_id,
                                    "component_pvalue": pvalue,
                                    "n_edges": int(size),
                                    "n_parcels": len(
                                        np.unique(
                                            np.r_[edge_i[selected], edge_j[selected]]
                                        )
                                    ),
                                    "t_min": component_t.min(),
                                    "t_max": component_t.max(),
                                    "direction": direction,
                                    "significant": bool(pvalue < NBS_ALPHA),
                                    "status": "components_found",
                                }
                            )
                        if not component_rows:
                            component_rows = [
                                {
                                    **common,
                                    "component_id": np.nan,
                                    "component_pvalue": np.nan,
                                    "n_edges": 0,
                                    "n_parcels": 0,
                                    "t_min": np.nan,
                                    "t_max": np.nan,
                                    "direction": "None",
                                    "significant": False,
                                    "status": "no_suprathreshold_components",
                                }
                            ]
                        components = pd.DataFrame(component_rows)
                        active = edge_labels > 0
                        edges = pd.DataFrame(
                            {
                                "roi_i": np.asarray(parcel_names)[edge_i[active]],
                                "roi_j": np.asarray(parcel_names)[edge_j[active]],
                                "component_id": edge_labels[active],
                                "component_pvalue": pvalues[edge_labels[active] - 1],
                                "t_value": observed_t[active],
                            }
                        )
                        for name, value in common.items():
                            edges[name] = value
                        edges["significant"] = edges["component_pvalue"].lt(NBS_ALPHA)
                        significant_edges = edges.loc[edges["significant"]]
                        adjacency = np.zeros((n_rois, n_rois), dtype=np.int32)
                        adjacency[edge_i, edge_j] = edge_labels
                        adjacency += adjacency.T
                        pd.DataFrame(
                            t_matrix, index=parcel_names, columns=parcel_names
                        ).to_csv(test_dir / f"edgewise_t_matrix_{stem}.csv")
                        components.to_csv(
                            test_dir / f"nbs_components_{stem}.csv", index=False
                        )
                        edges.to_csv(test_dir / f"nbs_edges_{stem}.csv", index=False)
                        significant_edges.to_csv(
                            test_dir / f"nbs_edges_significant_{stem}.csv", index=False
                        )
                        np.save(test_dir / f"nbs_adj_{stem}.npy", adjacency)
                        np.save(test_dir / f"nbs_null_{stem}.npy", null)
                        np.save(test_dir / f"nbs_component_pvals_{stem}.npy", pvalues)
                        all_components.append(components)
                        all_edges.append(edges)
                pd.concat(all_components, ignore_index=True).to_csv(
                    output_dir / "nbs_components_all_pairs_thresholds.csv", index=False
                )
                pd.concat(all_edges, ignore_index=True).to_csv(
                    output_dir / "nbs_edges_all_pairs_thresholds.csv", index=False
                )
                pd.concat(all_participants, ignore_index=True).to_csv(
                    nbs_dir / "nbs_included_participants.csv", index=False
                )
                pd.DataFrame(
                    all_missing,
                    columns=list(metadata.columns)
                    + ["con_type", "freq_band", "reason", "path"],
                ).to_csv(nbs_dir / "nbs_missing_matrices.csv", index=False)
        return (
            pd.concat(all_components, ignore_index=True),
            pd.concat(all_edges, ignore_index=True),
        )

    metadata = pd.read_csv(spectral_dir / "subject_info.csv", dtype={"subject": str})
    run_three_group_nbs(metadata, derivpath, load_parcels(), comparison_dir)


if __name__ == "__main__":
    main()
