#!/usr/bin/env python3
"""Generate golden test data for the Go HDBSCAN implementation.

Runs the reference HDBSCAN library on known datasets with various configurations
and exports results as JSON files for use in Go golden tests. Also benchmarks
the Python implementation for speed comparison.

Usage:
    uv run --with hdbscan --with scikit-learn --with numpy python3 testdata/generate_golden.py
"""

import json
import os
import time

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
    # Higher-dimensional dataset: 150 points in 10D, 3 clusters
    blobs10d_X, _ = make_blobs(n_samples=150, centers=3, n_features=10, random_state=42)
    # High-dimensional dataset: 100 points in 50D, 3 clusters
    blobs50d_X, _ = make_blobs(n_samples=100, centers=3, n_features=50, random_state=42)
    return {
        "blobs": blobs_X,
        "moons": moons_X,
        "small": small_X,
        "blobs10d": blobs10d_X,
        "blobs50d": blobs50d_X,
    }


# --- Configurations ---

CONFIGS = [
    # Original configs (Euclidean)
    {"min_cluster_size": 5, "cluster_selection_method": "eom", "allow_single_cluster": False, "metric": "euclidean"},
    {"min_cluster_size": 10, "cluster_selection_method": "eom", "allow_single_cluster": False, "metric": "euclidean"},
    {"min_cluster_size": 5, "cluster_selection_method": "leaf", "allow_single_cluster": False, "metric": "euclidean"},
    {"min_cluster_size": 5, "cluster_selection_method": "eom", "allow_single_cluster": True, "metric": "euclidean"},
    # Manhattan (L1)
    {"min_cluster_size": 5, "cluster_selection_method": "eom", "allow_single_cluster": False, "metric": "manhattan"},
    # Chebyshev (L-inf)
    {"min_cluster_size": 5, "cluster_selection_method": "eom", "allow_single_cluster": False, "metric": "chebyshev"},
    # Minkowski p=1.5
    {"min_cluster_size": 5, "cluster_selection_method": "eom", "allow_single_cluster": False, "metric": "minkowski", "p": 1.5},
    # Cosine
    {"min_cluster_size": 5, "cluster_selection_method": "eom", "allow_single_cluster": False, "metric": "cosine"},
]

# Which datasets get which configs
DATASET_CONFIGS = {
    # Original datasets keep original Euclidean configs
    "small": [c for c in CONFIGS if c["metric"] == "euclidean"],
    "blobs": [c for c in CONFIGS if c["metric"] == "euclidean"],
    "moons": [c for c in CONFIGS if c["metric"] == "euclidean"],
    # New datasets: Euclidean baseline + all non-Euclidean
    "blobs10d": CONFIGS,
    "blobs50d": CONFIGS,
}


def config_tag(cfg):
    """Create a short tag string from config for filename."""
    parts = [
        f"mcs{cfg['min_cluster_size']}",
        cfg["cluster_selection_method"],
    ]
    if cfg["allow_single_cluster"]:
        parts.append("single")
    if cfg["metric"] != "euclidean":
        metric_tag = cfg["metric"]
        if cfg["metric"] == "minkowski" and "p" in cfg:
            metric_tag += str(cfg["p"]).replace(".", "")
        parts.append(metric_tag)
    return "_".join(parts)


def run_hdbscan(data, cfg):
    """Run HDBSCAN with the given config and return results dict."""
    kwargs = {
        "min_cluster_size": cfg["min_cluster_size"],
        "cluster_selection_method": cfg["cluster_selection_method"],
        "allow_single_cluster": cfg["allow_single_cluster"],
        "metric": cfg["metric"],
        "algorithm": "generic",  # brute-force for reproducibility
    }
    if cfg["metric"] == "minkowski" and "p" in cfg:
        kwargs["p"] = cfg["p"]

    clusterer = hdbscan.HDBSCAN(**kwargs)
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


def generate_golden_files(datasets):
    """Generate all golden test JSON files."""
    count = 0
    for dataset_name, data in datasets.items():
        configs = DATASET_CONFIGS.get(dataset_name, CONFIGS)
        for cfg in configs:
            tag = config_tag(cfg)
            filename = f"{dataset_name}_{tag}.json"
            filepath = os.path.join(OUTDIR, filename)

            print(f"Generating {filename}...")

            results = run_hdbscan(data, cfg)

            config_out = {
                "min_cluster_size": cfg["min_cluster_size"],
                "cluster_selection_method": cfg["cluster_selection_method"],
                "allow_single_cluster": cfg["allow_single_cluster"],
                "metric": cfg["metric"],
            }
            if cfg["metric"] == "minkowski" and "p" in cfg:
                config_out["p"] = cfg["p"]

            output = {
                "dataset": dataset_name,
                "config": config_out,
                "data": data.tolist(),
                **results,
            }

            with open(filepath, "w") as f:
                json.dump(output, f, indent=2)

            n_clusters = len(set(results["labels"])) - (1 if -1 in results["labels"] else 0)
            n_noise = results["labels"].count(-1)
            print(f"  -> {n_clusters} clusters, {n_noise} noise points")
            count += 1

    print(f"\n{count} golden test files written to {OUTDIR}/")


def run_benchmarks():
    """Benchmark Python HDBSCAN on various dataset sizes for speed comparison."""
    print("\n" + "=" * 60)
    print("BENCHMARKS: Python HDBSCAN (brute-force, Euclidean)")
    print("=" * 60)

    rng = np.random.RandomState(42)
    sizes = [100, 500, 1000, 5000]
    dims = 2
    n_runs = 5

    bench_results = {}

    for n in sizes:
        data = rng.rand(n, dims) * 100
        times = []

        # Warm-up run
        hdbscan.HDBSCAN(min_cluster_size=5, algorithm="generic", metric="euclidean").fit(data)

        for _ in range(n_runs):
            start = time.perf_counter()
            hdbscan.HDBSCAN(min_cluster_size=5, algorithm="generic", metric="euclidean").fit(data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg = np.mean(times)
        std = np.std(times)
        bench_results[f"brute_euclidean_{n}pt_2d"] = {
            "n": n,
            "dims": dims,
            "metric": "euclidean",
            "algorithm": "generic",
            "runs": n_runs,
            "mean_seconds": float(avg),
            "std_seconds": float(std),
            "times_seconds": [float(t) for t in times],
        }
        print(f"  n={n:>5}, dims={dims}: {avg*1000:>10.2f} ms (±{std*1000:.2f} ms)")

    # Also benchmark with best algorithm (tree-accelerated) for comparison
    print("\nBENCHMARKS: Python HDBSCAN (best algorithm, Euclidean)")
    print("-" * 60)

    for n in sizes:
        data = rng.rand(n, dims) * 100
        times = []

        # Warm-up
        hdbscan.HDBSCAN(min_cluster_size=5, algorithm="best", metric="euclidean").fit(data)

        for _ in range(n_runs):
            start = time.perf_counter()
            hdbscan.HDBSCAN(min_cluster_size=5, algorithm="best", metric="euclidean").fit(data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg = np.mean(times)
        std = np.std(times)
        bench_results[f"best_euclidean_{n}pt_2d"] = {
            "n": n,
            "dims": dims,
            "metric": "euclidean",
            "algorithm": "best",
            "runs": n_runs,
            "mean_seconds": float(avg),
            "std_seconds": float(std),
            "times_seconds": [float(t) for t in times],
        }
        print(f"  n={n:>5}, dims={dims}: {avg*1000:>10.2f} ms (±{std*1000:.2f} ms)")

    # High-dimensional benchmarks
    print("\nBENCHMARKS: Python HDBSCAN (brute-force, Euclidean, high-dim)")
    print("-" * 60)

    for n, d in [(100, 10), (100, 50), (500, 10), (500, 50)]:
        data = rng.rand(n, d) * 100
        times = []

        hdbscan.HDBSCAN(min_cluster_size=5, algorithm="generic", metric="euclidean").fit(data)

        for _ in range(n_runs):
            start = time.perf_counter()
            hdbscan.HDBSCAN(min_cluster_size=5, algorithm="generic", metric="euclidean").fit(data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg = np.mean(times)
        std = np.std(times)
        bench_results[f"brute_euclidean_{n}pt_{d}d"] = {
            "n": n,
            "dims": d,
            "metric": "euclidean",
            "algorithm": "generic",
            "runs": n_runs,
            "mean_seconds": float(avg),
            "std_seconds": float(std),
            "times_seconds": [float(t) for t in times],
        }
        print(f"  n={n:>5}, dims={d:>2}: {avg*1000:>10.2f} ms (±{std*1000:.2f} ms)")

    # Non-Euclidean metric benchmarks
    print("\nBENCHMARKS: Python HDBSCAN (brute-force, Manhattan)")
    print("-" * 60)

    for n in [100, 500, 1000]:
        data = rng.rand(n, dims) * 100
        times = []

        hdbscan.HDBSCAN(min_cluster_size=5, algorithm="generic", metric="manhattan").fit(data)

        for _ in range(n_runs):
            start = time.perf_counter()
            hdbscan.HDBSCAN(min_cluster_size=5, algorithm="generic", metric="manhattan").fit(data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg = np.mean(times)
        std = np.std(times)
        bench_results[f"brute_manhattan_{n}pt_2d"] = {
            "n": n,
            "dims": 2,
            "metric": "manhattan",
            "algorithm": "generic",
            "runs": n_runs,
            "mean_seconds": float(avg),
            "std_seconds": float(std),
            "times_seconds": [float(t) for t in times],
        }
        print(f"  n={n:>5}, dims={dims}: {avg*1000:>10.2f} ms (±{std*1000:.2f} ms)")

    # Write benchmark results
    bench_path = os.path.join(OUTDIR, "python_benchmarks.json")
    with open(bench_path, "w") as f:
        json.dump(bench_results, f, indent=2)
    print(f"\nBenchmark results written to {bench_path}")


def main():
    datasets = make_datasets()
    generate_golden_files(datasets)
    run_benchmarks()


if __name__ == "__main__":
    main()
