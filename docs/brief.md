# Project Brief: hdbscan-go

## What

A native Go implementation of the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) algorithm, ported from the [scikit-learn-contrib/hdbscan](https://github.com/scikit-learn-contrib/hdbscan) Python library.

## Why

Existing Go implementations (Belval/hdbscan, humilityai/hdbscan) are incomplete and use naive O(n²) MST construction. There is no production-quality Go library that provides the full HDBSCAN feature set with competitive performance. A proper implementation would make HDBSCAN accessible to Go-based data pipelines, real-time services, and systems where Python is impractical.

## Reference Material

- **Source library:** `reference/scikit-learn-contrib-hdbscan/` — the full Python/Cython codebase
- **Key paper:** McInnes, Healy, & Astels. "hdbscan: Hierarchical density based clustering" (JOSS, 2017)
- **Core algorithm papers:** Campello et al. 2013, 2015 (density-based clustering hierarchy + GLOSH outlier detection)
- **Initial analysis:** `docs/initial-convo.md` — module-by-module breakdown with Go design notes

## Algorithm Pipeline

HDBSCAN runs as a 6-stage pipeline:

1. **Core distance computation** — k-nearest-neighbor distances via spatial trees
2. **Mutual reachability graph** — transform pairwise distances using core distances
3. **Minimum spanning tree** — Prim's (brute) or Borůvka's (tree-accelerated) on the reachability graph
4. **Single-linkage tree** — union-find to build dendrogram from sorted MST edges
5. **Condensed tree** — prune single-linkage tree by `min_cluster_size`
6. **Cluster extraction** — excess-of-mass (EOM) or leaf selection on the condensed tree

Optional extensions: GLOSH outlier scores, soft clustering / membership vectors, approximate prediction for new points, FLASC branch detection.

## API Design

Go-idiomatic functional API with options pattern — no scikit-learn `fit`/`predict` paradigm:

```go
type Config struct {
    MinClusterSize         int
    MinSamples             int
    Metric                 DistanceMetric
    ClusterSelectionMethod string // "eom" or "leaf"
    Alpha                  float64
    AllowSingleCluster     bool
}

type Result struct {
    Labels        []int
    Probabilities []float64
    Stabilities   map[int]float64
}

func Cluster(data [][]float64, cfg Config) (*Result, error)
```

Data representation: flat `[]float64` with stride-based indexing for cache locality, or `gonum/mat.Dense` where convenient.

## Scope

### MVP (Phase 1)

| Component | Description |
|---|---|
| Distance metrics | Euclidean, Manhattan, Cosine, plus `DistanceFunc` interface for custom metrics |
| Mutual reachability | Brute-force pairwise distance + core distance transform |
| Prim's MST | Priority-queue-based MST on full reachability graph |
| Union-Find | Disjoint-set with path compression + union by rank |
| Condensed tree | Dendrogram pruning by `min_cluster_size` |
| EOM extraction | Excess-of-mass cluster selection with stability scores |
| GLOSH | Outlier scores derived from condensed tree |

**Target:** Working, correct clustering with O(n²) performance. Validate against Python reference output on standard test datasets.

### Optimization (Phase 2)

| Component | Description |
|---|---|
| KD-tree | With k-NN query support for low-dimensional data |
| Ball tree | With k-NN query support for higher dimensions / non-Euclidean metrics |
| Borůvka MST | Dual-tree accelerated MST construction — O(n log n) for low-d data |
| Goroutine parallelism | Parallel distance computation, k-NN queries, MST steps |

### Full Feature Parity (Phase 3)

| Component | Description |
|---|---|
| Additional distance metrics | Haversine, Mahalanobis, Hamming, etc. |
| Soft clustering | Membership vectors, all-points membership |
| Approximate prediction | Predict clusters for unseen points against fitted model |
| Leaf cluster selection | Alternative to EOM extraction |
| Robust single linkage | Thin wrapper over core pipeline |
| FLASC branch detection | Detect branches within clusters |
| Cluster validity | DBCV index for evaluating clustering quality |

### Out of Scope

- Visualization / plotting (no Go equivalent needed)
- scikit-learn API compatibility
- Sparse matrix input (can revisit later)

## Key Design Decisions

- **Start from the algorithm, not the Python code.** Translate the math, not the NumPy idioms.
- **`gonum` as foundation** for matrix ops and linear algebra where it makes sense.
- **Goroutine-friendly from the start** — design data structures for safe concurrent access even if Phase 1 is single-threaded.
- **Test against reference implementation** — generate ground-truth outputs from the Python library on known datasets, use those as golden tests in Go.

## Estimated Size

- Phase 1 (MVP): ~2,000–3,000 lines of Go
- Phase 2 (optimized): ~4,000–6,000 lines
- Phase 3 (full parity): ~7,000–9,000 lines
