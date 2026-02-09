# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Go implementation of HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise), ported from the [scikit-learn-contrib/hdbscan](https://github.com/scikit-learn-contrib/hdbscan) Python library. The reference source is in `reference/scikit-learn-contrib-hdbscan/` (key files: `_hdbscan_tree.pyx`, `_hdbscan_linkage.pyx`, `_hdbscan_boruvka.pyx`).

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

Single `hdbscan` package. `Cluster()` in `hdbscan.go` dispatches to one of three pipeline paths based on `Config.Algorithm`:

### Brute-force path (`AlgorithmBrute`)

```
Input data → Cluster(data, cfg)
  → Pairwise distances          (distance.go)
  → Core distances               (core_distance.go)
  → Mutual reachability graph    (reachability.go)
  → Prim's MST on full matrix   (mst.go)
  → clusterFromMST              (hdbscan.go — shared post-MST pipeline)
```

### Prim's tree path (`AlgorithmPrimsKDTree`, `AlgorithmPrimsBalltree`)

```
Input data → Cluster(data, cfg)
  → Build spatial tree           (kdtree.go or balltree.go)
  → Tree core distances          (tree_core_distance.go)
  → Prim's MST on-the-fly       (mst_vector.go — O(n) memory, no full matrix)
  → clusterFromMST
```

### Borůvka tree path (`AlgorithmBoruvkaKDTree`, `AlgorithmBoruvkaBalltree`)

```
Input data → Cluster(data, cfg)
  → Build spatial tree           (kdtree.go or balltree.go)
  → Borůvka dual-tree MST       (boruvka.go — tree KNN for core distances internally)
  → clusterFromMST
```

### Shared post-MST pipeline (`clusterFromMST`)

```
  MST edges
  → Single-linkage dendrogram    (unionfind.go, label.go)
  → Condensed tree               (condensed_tree.go)
  → Stability computation        (stability.go)
  → Cluster selection: EOM/leaf  (cluster_selection.go)
  → Labels + probabilities       (labeling.go)
  → GLOSH outlier scores         (outlier.go)
```

Optional: `SimplifyHierarchy` in `simplify.go` prunes low-persistence clusters before selection.

### Algorithm selection (`algorithm.go`)

`selectAlgorithm()` resolves `AlgorithmAuto`: custom metric → brute, KD-tree-valid metric + dims ≤ 60 → `boruvka_kdtree`, otherwise → `boruvka_balltree`. `ClusterPrecomputed()` always uses brute (no raw data for trees).

### Entry points (`hdbscan.go`)

- `Cluster(data [][]float64, cfg Config) (*Result, error)` — from raw data
- `ClusterPrecomputed(distMatrix []float64, n int, cfg Config) (*Result, error)` — from precomputed distance matrix

## Shared Helpers

`cluster_selection.go` defines shared helpers reused across the selection/output layer:
- `treeRoot(tree)` — finds root cluster ID (smallest parent); used by `outlier.go` and `labeling.go`
- `clusterEntries(tree)` — filters to cluster-to-cluster entries (`ChildSize > 1`)
- `clusterChildrenMap(tree)` — builds parent→children mapping
- `bfsDescendants(childrenOf, root)` — BFS traversal from a cluster node

## Key Design Decisions

- **Flat `[]float64` with stride-based indexing** for matrices and tree data (row-major). Access: `matrix[i*n + j]`. Cache locality matters for O(n²) distance computation.
- **`DistanceMetric` interface** with `Distance()`, `ReducedDistance()`, `DistToRdist()`, and `RdistToDist()` methods. Scalar conversion methods enable spatial tree pruning (e.g., Euclidean: `DistToRdist(d) = d*d`, `RdistToDist(d) = sqrt(d)`; most metrics: identity). `DistanceFunc` adapter wraps simple functions.
- **`SpatialTree` / `BoruvkaTree` interfaces** (`spatial_tree.go`) decouple tree implementations from algorithms. Both `KDTree` and `BallTree` implement both interfaces. `BoruvkaTree` adds `MinRdistDual`/`MinRdistPoint` for dual-tree pruning.
- **MST uses chain format**: edges are `(from, to, weight)`. For `PrimMST` (matrix), `from` is the previously-added node. For `PrimMSTVector` (on-the-fly), `from` is the actual nearest tree neighbor via `currentSources[]`.
- **Borůvka operates in true distance space** — tree's `MinRdistDual` returns rdist, converted to true distance via `metric.RdistToDist()` before comparing with bounds. Simpler than the reference (which uses rdist for KD-tree and dist for Ball tree).
- **GLOSH outlier scores use per-cluster maxLambda without upward propagation**. Each cluster's death value is the max lambda among its direct condensed-tree children only.
- **No hidden mutable shared state** during computation — goroutine-safe by design.

## Data Structures

- **Distance/reachability matrices**: flat `[]float64`, length n×n, row-major.
- **MST edges**: `[][3]float64` — each edge is `[from, to, weight]`.
- **Single-linkage dendrogram**: `[][4]float64` — scipy format: `[left, right, distance, size]`. Internal node IDs start at n.
- **Condensed tree**: `[]CondensedTreeEntry` — `{Parent, Child int; LambdaVal float64; ChildSize int}`. Lambda = 1/distance. Entries with `Child < rootCluster` are points; entries with `Child >= rootCluster` are sub-clusters.
- **Spatial trees** (`KDTree`, `BallTree`): binary heap layout (children at `2*i+1`, `2*i+2`). Point data reordered via `idxArray` permutation. KD-tree stores per-node axis-aligned bounds (`nodeBoundsMin`/`nodeBoundsMax`). Ball tree stores per-node centroid + radius with precomputed centroid distance matrix.
- **Union-Find** (`unionfind.go`): path compression + union by size, 2n-1 element capacity. Used in dendrogram construction (`label.go`) and cluster labeling (`labeling.go`). Borůvka has its own separate `boruvkaUnionFind` in `boruvka.go`.

## Testing

**Golden tests** (`testdata/`): 12 JSON files (3 datasets × 4 configs) generated from the Python reference via `testdata/generate_golden.py`. Golden tests force `AlgorithmBrute` since the golden data was generated by the brute-force path. Four test groups in `golden_test.go`:

1. `TestGoldenSmallStrict` — strict 1e-10 tolerance on the 6-point `small` dataset (no equal-weight MST edges, so the dendrogram is deterministic).
2. `TestGoldenLabels` — permutation-invariant label comparison. Skips leaf selection on `blobs`/`moons` datasets (see Known Limitations).
3. `TestGoldenProbsAndScores` — relaxed 0.2 tolerance for probabilities and outlier scores on larger datasets. Skips leaf selection on non-small datasets.
4. `TestGoldenAlgorithms` — cross-algorithm comparison: all 5 algorithm paths produce equivalent labels on the small dataset (20 subtests).

**Algorithm equivalence tests** (`hdbscan_test.go`):
- `TestAlgorithmEquivalence` — 50-point dataset, EOM selection, verifies labels/probabilities/outlier scores match brute-force for all tree-based algorithms.
- `TestAlgorithmEquivalenceLeaf` — 50-point dataset, leaf selection, verifies structural properties (cluster count, group cohesion) across all algorithms.

**Unit tests**: each pipeline stage has its own `*_test.go`. Edge cases cover n=1, n=2, all-identical points, minClusterSize > n, +Inf distances. Borůvka tests use a `bruteForceBoruvkaTree` mock implementing `BoruvkaTree` for algorithm-level testing independent of real tree implementations.

## Known Limitations

**Dendrogram sort tie-breaking**: when MST edges have equal weights, Go's `sort.Slice` and NumPy's `argsort(kind='mergesort')` produce different orderings. This yields different (but equally valid) dendrograms, condensed trees, and downstream results for a small number of boundary points. The `small` dataset has no equal-weight edges and matches exactly. Larger datasets use relaxed tolerances. Leaf cluster selection is particularly sensitive to this since it depends on the exact condensed tree structure.

**Algorithm path divergence**: different MST construction algorithms (brute Prim's, vector Prim's, Borůvka) can produce MSTs with different edge orderings when edge weights tie. This causes downstream differences in condensed tree structure and leaf selection, but EOM selection on well-separated data is stable across all paths.

## Documentation

- `docs/brief.md` — project overview and scope
- `docs/requirements.md` — functional/non-functional requirements (FR/NFR numbering)
- `docs/tasks.md` — implementation task breakdown with dependencies
- `docs/initial-convo.md` — module-by-module analysis of the Python reference
