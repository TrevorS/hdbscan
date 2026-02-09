#!/usr/bin/env python3
"""Generate golden test data for the Go HDBSCAN implementation.

Runs the reference HDBSCAN library on known datasets with various configurations
and exports results as JSON files for use in Go golden tests.

Usage:
    uv run --with hdbscan --with scikit-learn --with numpy python3 testdata/generate_golden.py
"""

import json
import os

import hdbscan
import numpy as np
from hdbscan._hdbscan_tree import compute_stability
from sklearn.datasets import make_blobs, make_moons

# Output directory
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# --- Datasets ---

def make_datasets():
    """Generate the test datasets."""
    blobs_X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    moons_X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    small_X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [10.0, 10.0],
        [11.0, 10.0],
        [10.0, 11.0],
    ])
    return {
        "blobs": blobs_X,
        "moons": moons_X,
        "small": small_X,
    }


# --- Configurations ---

CONFIGS = [
    {"min_cluster_size": 5, "cluster_selection_method": "eom", "allow_single_cluster": False},
    {"min_cluster_size": 10, "cluster_selection_method": "eom", "allow_single_cluster": False},
    {"min_cluster_size": 5, "cluster_selection_method": "leaf", "allow_single_cluster": False},
    {"min_cluster_size": 5, "cluster_selection_method": "eom", "allow_single_cluster": True},
]


def config_tag(cfg):
    """Create a short tag string from config for filename."""
    parts = [
        f"mcs{cfg['min_cluster_size']}",
        cfg["cluster_selection_method"],
    ]
    if cfg["allow_single_cluster"]:
        parts.append("single")
    return "_".join(parts)


def run_hdbscan(data, cfg):
    """Run HDBSCAN with the given config and return results dict."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        cluster_selection_method=cfg["cluster_selection_method"],
        allow_single_cluster=cfg["allow_single_cluster"],
        metric="euclidean",
        algorithm="generic",  # brute-force for reproducibility
    )
    clusterer.fit(data)

    # Extract condensed tree as list of dicts
    ct = clusterer.condensed_tree_._raw_tree
    condensed_tree = []
    for i in range(len(ct)):
        condensed_tree.append({
            "parent": int(ct["parent"][i]),
            "child": int(ct["child"][i]),
            "lambda_val": float(ct["lambda_val"][i]),
            "child_size": int(ct["child_size"][i]),
        })

    # Extract single linkage tree as list of [left, right, distance, size]
    slt = clusterer.single_linkage_tree_.to_numpy()
    single_linkage_tree = []
    for row in slt:
        single_linkage_tree.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])

    # Compute stabilities from raw condensed tree
    # Keys are floats (cluster IDs), convert to string for JSON
    stabilities = {}
    for k, v in compute_stability(ct).items():
        stabilities[str(int(k))] = float(v)

    return {
        "labels": [int(x) for x in clusterer.labels_],
        "probabilities": [float(x) for x in clusterer.probabilities_],
        "outlier_scores": [float(x) for x in clusterer.outlier_scores_],
        "stabilities": stabilities,
        "condensed_tree": condensed_tree,
        "single_linkage_tree": single_linkage_tree,
    }


def main():
    datasets = make_datasets()

    for dataset_name, data in datasets.items():
        for cfg in CONFIGS:
            tag = config_tag(cfg)
            filename = f"{dataset_name}_{tag}.json"
            filepath = os.path.join(OUTDIR, filename)

            print(f"Generating {filename}...")

            results = run_hdbscan(data, cfg)

            output = {
                "dataset": dataset_name,
                "config": {
                    "min_cluster_size": cfg["min_cluster_size"],
                    "cluster_selection_method": cfg["cluster_selection_method"],
                    "allow_single_cluster": cfg["allow_single_cluster"],
                    "metric": "euclidean",
                },
                "data": data.tolist(),
                **results,
            }

            with open(filepath, "w") as f:
                json.dump(output, f, indent=2)

            n_clusters = len(set(results["labels"])) - (1 if -1 in results["labels"] else 0)
            n_noise = results["labels"].count(-1)
            print(f"  -> {n_clusters} clusters, {n_noise} noise points")

    print(f"\nAll golden test data written to {OUTDIR}/")


if __name__ == "__main__":
    main()
