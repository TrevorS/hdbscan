# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Go implementation of HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise), ported from the [scikit-learn-contrib/hdbscan](https://github.com/scikit-learn-contrib/hdbscan) Python library. The reference source is in `reference/scikit-learn-contrib-hdbscan/` (key files: `_hdbscan_tree.pyx`, `_hdbscan_linkage.pyx`).

## Build & Test Commands

```bash
go build ./...                    # build
go test ./...                     # all tests
go test -run TestName ./...       # single test
go test -bench=. -benchmem ./...  # benchmarks
go test -race ./...               # race detector
golangci-lint run                 # lint
gofmt -w .                        # format
```

Makefile shortcuts: `make test`, `make bench`, `make lint`, `make fmt`, `make build`, `make race`, `make cover`.

## Architecture

Single `hdbscan` package. The pipeline flows through these stages, each feeding the next:

```
Input data  →  Cluster(data, cfg) in hdbscan.go
  → Pairwise distances          (distance.go, core_distance.go)
  → Core distances               (core_distance.go)
  → Mutual reachability graph    (reachability.go)
  → Minimum spanning tree        (mst.go — Prim's, chain format)
  → Single-linkage dendrogram    (unionfind.go, label.go)
  → Condensed tree               (condensed_tree.go)
  → Stability computation        (stability.go)
  → Cluster selection: EOM/leaf  (cluster_selection.go)
  → Labels + probabilities       (labeling.go)
  → GLOSH outlier scores         (outlier.go)
```

Optional: `SimplifyHierarchy` in `simplify.go` prunes low-persistence clusters before selection.

Two entry points in `hdbscan.go`:
- `Cluster(data [][]float64, cfg Config) (*Result, error)` — from raw data
- `ClusterPrecomputed(distMatrix []float64, n int, cfg Config) (*Result, error)` — from precomputed distance matrix

## Shared Helpers

`cluster_selection.go` defines shared helpers reused across the selection/output layer:
- `treeRoot(tree)` — finds root cluster ID (smallest parent); used by `outlier.go` and `labeling.go`
- `clusterEntries(tree)` — filters to cluster-to-cluster entries (`ChildSize > 1`)
- `clusterChildrenMap(tree)` — builds parent→children mapping
- `bfsDescendants(childrenOf, root)` — BFS traversal from a cluster node

## Key Design Decisions

- **Flat `[]float64` with stride-based indexing** for matrices (row-major), not `[][]float64`. Access: `matrix[i*n + j]`. Cache locality matters for O(n²) distance computation.
- **`DistanceMetric` interface** with `Distance()` and `ReducedDistance()` methods. `DistanceFunc` adapter wraps simple functions. Built-in metrics: Euclidean, Manhattan, Cosine, Chebyshev, Minkowski.
- **MST uses chain format**: edges are `(currentNode, newNode, weight)` where `currentNode` is the previously-added node, matching the reference `mst_linkage_core` output.
- **GLOSH outlier scores use per-cluster maxLambda without upward propagation**. Each cluster's death value is the max lambda among its direct condensed-tree children only.
- **No hidden mutable shared state** during computation — goroutine-safe by design.

## Data Structures

- **Distance/reachability matrices**: flat `[]float64`, length n×n, row-major.
- **MST edges**: `[][3]float64` — each edge is `[from, to, weight]`.
- **Single-linkage dendrogram**: `[][4]float64` — scipy format: `[left, right, distance, size]`. Internal node IDs start at n.
- **Condensed tree**: `[]CondensedTreeEntry` — `{Parent, Child int; LambdaVal float64; ChildSize int}`. Lambda = 1/distance. Entries with `Child < rootCluster` are points; entries with `Child >= rootCluster` are sub-clusters.
- **Union-Find** (`unionfind.go`): path compression + union by size, 2n-1 element capacity. Used in both dendrogram construction (label.go) and cluster labeling (labeling.go, which has its own private union-find).

## Testing

**Golden tests** (`testdata/`): 12 JSON files (3 datasets × 4 configs) generated from the Python reference via `testdata/generate_golden.py`. Three test tiers in `golden_test.go`:

1. `TestGoldenSmallStrict` — strict 1e-10 tolerance on the 6-point `small` dataset (no equal-weight MST edges, so the dendrogram is deterministic).
2. `TestGoldenLabels` — permutation-invariant label comparison. Skips leaf selection on `blobs`/`moons` datasets (see Known Limitations).
3. `TestGoldenProbsAndScores` — relaxed 0.2 tolerance for probabilities and outlier scores on larger datasets. Skips leaf selection on non-small datasets.

**Unit tests**: each pipeline stage has its own `*_test.go`. Edge cases cover n=1, n=2, all-identical points, minClusterSize > n, +Inf distances.

## Known Limitations

**Dendrogram sort tie-breaking**: when MST edges have equal weights, Go's `sort.Slice` and NumPy's `argsort(kind='mergesort')` produce different orderings. This yields different (but equally valid) dendrograms, condensed trees, and downstream results for a small number of boundary points. The `small` dataset has no equal-weight edges and matches exactly. Larger datasets use relaxed tolerances.

## Documentation

- `docs/brief.md` — project overview and scope
- `docs/requirements.md` — functional/non-functional requirements (FR/NFR numbering)
- `docs/tasks.md` — implementation task breakdown with dependencies
- `docs/initial-convo.md` — module-by-module analysis of the Python reference
