# Requirements: hdbscan-go

Requirements extracted from `docs/brief.md`, `docs/initial-convo.md`, and the
`reference/scikit-learn-contrib-hdbscan/` Python/Cython source.

---

## Functional Requirements

### FR-1: Distance Metrics

**FR-1.1: Built-in Euclidean Distance**
Compute the L2 (Euclidean) distance between two float64 vectors.
- Acceptance: `dist(a, b) == sqrt(sum((a[i]-b[i])^2))` for all i.
- Must also support a "reduced" (squared) form for tree-pruning optimizations.

**FR-1.2: Built-in Manhattan Distance**
Compute the L1 (Manhattan / cityblock) distance.
- Acceptance: `dist(a, b) == sum(|a[i]-b[i]|)`.

**FR-1.3: Built-in Cosine / Arccos Distance**
Compute cosine dissimilarity (1 - cosine similarity), aliased as "arccos".
- Acceptance: matches `scipy.spatial.distance.cosine` within float64 tolerance.

**FR-1.4: Built-in Chebyshev Distance**
Compute the L-infinity (max absolute difference) distance.
- Acceptance: `dist(a, b) == max(|a[i]-b[i]|)`.

**FR-1.5: Built-in Minkowski Distance**
Compute the Lp distance parameterized by a `p` value.
- When p=2 must equal Euclidean; when p=1 must equal Manhattan.
- Must reject negative p values.

**FR-1.6: DistanceFunc / DistanceMetric Interface**
Expose both:
- A simple function type: `type DistanceFunc func(a, b []float64) float64`
- An interface with `Distance` and optionally `ReducedDistance` methods for tree optimizations.

Users must be able to supply custom metrics through either mechanism.

**FR-1.7: Precomputed Distance Matrix Input**
Accept a precomputed symmetric distance matrix as input instead of raw feature vectors.
- Must support `math.Inf` values to indicate missing distances.
- Must validate that the matrix is square.

**FR-1.8: Additional Distance Metrics (Phase 3)**
Haversine, Mahalanobis, Hamming, Canberra, Bray-Curtis, Jaccard, Dice,
Standardized Euclidean, Weighted Minkowski, Rogers-Tanimoto, Russell-Rao,
Sokal-Michener, Sokal-Sneath.
- Each must conform to the DistanceMetric interface.
- Dependencies: FR-1.6.

---

### FR-2: Core Distance Computation

**FR-2.1: Brute-Force Core Distances**
Given an n x n distance matrix and a `min_samples` parameter, compute the core
distance for each point as the distance to its `min_samples`-th nearest neighbor
(using partial sort / nth-element selection, not full sort).
- Acceptance: core_distances[i] == sorted(distance_matrix[i])[min_samples] for
  each point i.
- `min_samples` is clamped to `min(n-1, min_samples)` when it exceeds the
  dataset size.

**FR-2.2: KD-Tree k-NN Core Distances (Phase 2)**
Build a KD-tree from the input data and use k-NN queries to compute core
distances without materializing the full distance matrix.
- Acceptance: produces identical core distances to FR-2.1 for Euclidean and
  other KD-tree-valid metrics.
- Dependencies: FR-3.1.

**FR-2.3: Ball Tree k-NN Core Distances (Phase 2)**
Build a ball tree from the input data and use k-NN queries for core distances.
- Supports all metrics valid for ball trees (superset of KD-tree metrics).
- Dependencies: FR-3.2.

---

### FR-3: Spatial Index Trees (Phase 2)

**FR-3.1: KD-Tree**
Implement a KD-tree with:
- Construction from n-dimensional point data.
- k-nearest-neighbor query returning indices and distances.
- Configurable leaf size (default 40).
- Valid metrics: Euclidean, L2, Minkowski, Manhattan, Cityblock, L1, Chebyshev.

**FR-3.2: Ball Tree**
Implement a ball tree with:
- Construction from n-dimensional point data.
- k-nearest-neighbor query returning indices and distances.
- Configurable leaf size (default 40).
- Valid metrics: all KD-tree metrics plus Haversine, Mahalanobis, Bray-Curtis,
  Canberra, Hamming, Jaccard, Dice, etc.

**FR-3.3: Dual-Tree Traversal (Phase 2)**
Support dual-tree query patterns required by the Boruvka MST algorithm.
- Dependencies: FR-3.1 or FR-3.2.

---

### FR-4: Mutual Reachability Graph

**FR-4.1: Dense Mutual Reachability**
Given a distance matrix, core distances, and an `alpha` scaling parameter,
compute the mutual reachability distance matrix:
```
mr_dist(a, b) = max(core_dist(a), core_dist(b), dist(a,b) / alpha)
```
- When `alpha == 1.0`, skip the division.
- Acceptance: element-wise match with reference `mutual_reachability()` output.
- Dependencies: FR-2.1.

**FR-4.2: Implicit Mutual Reachability in Prim's MST**
When using `mst_linkage_core_vector`, mutual reachability distances are computed
on-the-fly inside the MST loop rather than materializing the full matrix.
- Acceptance: the MST produced is identical to the one from FR-4.1 + FR-5.1.
- Dependencies: FR-2.2 or FR-2.3, FR-5.2.

---

### FR-5: Minimum Spanning Tree Construction

**FR-5.1: Prim's MST on Dense Matrix**
Implement Prim's algorithm on a full mutual reachability distance matrix.
- Input: n x n float64 mutual reachability matrix.
- Output: (n-1) x 3 array of [from, to, weight] edges.
- Warn (do not error) if any MST edge weight is +Inf (indicates missing
  distances in the input).
- Dependencies: FR-4.1.

**FR-5.2: Prim's MST with Core Distance Vector (Phase 2)**
Implement `mst_linkage_core_vector`: Prim's MST that takes raw data, core
distances, and a distance metric, computing mutual reachability on-the-fly.
- Avoids allocating the full n x n distance matrix.
- The `alpha` parameter scales raw distances by `1/alpha` before comparison.
- Dependencies: FR-1.6, FR-2.2 or FR-2.3.

**FR-5.3: Boruvka Dual-Tree MST (Phase 2)**
Implement the dual-tree Boruvka MST construction from McInnes & Healy (2017).
- Uses KD-tree or ball tree for sub-linear MST construction.
- Supports `approx_min_span_tree` flag (default true) for speed vs. exactness
  tradeoff.
- Dependencies: FR-3.3.

---

### FR-6: Single-Linkage Tree (Union-Find)

**FR-6.1: Union-Find Data Structure**
Implement a disjoint-set (union-find) with:
- Path compression in `Find`.
- Union by rank (or size).
- Tracking of component sizes.
- The `label` function: process sorted MST edges to produce a single-linkage
  hierarchy in scipy format (n-1 rows of [left, right, distance, size]).

**FR-6.2: Label Function**
Given an (n-1) x 3 MST edge array sorted by weight, produce a single-linkage
dendrogram as an (n-1) x 4 array: `[cluster_left, cluster_right, distance, merged_size]`.
- Cluster indices for merged nodes start at n and increment.
- Dependencies: FR-6.1.

---

### FR-7: Condensed Tree

**FR-7.1: Condense Tree**
Given a single-linkage dendrogram and `min_cluster_size`, produce a condensed
tree as a structured record array with fields:
`(parent int, child int, lambda_val float64, child_size int)`

Logic:
- Walk the dendrogram in BFS order from the root.
- `lambda_val = 1.0 / distance` (or +Inf if distance == 0).
- When both children of a split have size >= `min_cluster_size`, create two new
  cluster nodes.
- When both children are smaller, collapse all points into the parent cluster.
- When one child is smaller, keep the larger child as the same cluster and
  collapse the smaller.
- Dependencies: FR-6.2.

**FR-7.2: Compute Stability**
Given a condensed tree, compute a stability score for each cluster:
```
stability(C) = sum over points p in C: (lambda(p) - lambda_birth(C)) * size(p)
```
- `lambda_birth` of a cluster is the minimum lambda at which it first appears
  as a child.
- The root cluster has `lambda_birth = 0`.
- Return as `map[int]float64`.
- Dependencies: FR-7.1.

**FR-7.3: Simplify Hierarchy by Persistence**
Given a condensed tree and a `persistence_threshold`, remove leaf clusters
whose persistence (birth lambda difference from parent) is below the threshold.
Relabel remaining clusters for consecutive numbering.
- Dependencies: FR-7.1.

---

### FR-8: Cluster Extraction

**FR-8.1: Excess-of-Mass (EOM) Cluster Selection**
Walk the condensed tree bottom-up. For each non-leaf cluster, compare its own
stability to the sum of its children's stabilities:
- If children's total > parent's stability (or parent exceeds
  `max_cluster_size`, or parent's epsilon > `cluster_selection_epsilon_max`):
  mark parent as NOT a cluster, update its stability to children's sum.
- Otherwise: mark parent as a cluster, mark all descendants as NOT clusters.

Then apply epsilon-based merging if `cluster_selection_epsilon > 0`:
- Traverse upward from each selected cluster; if the cluster's epsilon
  (1/lambda) is below the threshold, merge into ancestor.

- Dependencies: FR-7.2, FR-8.3.

**FR-8.2: Leaf Cluster Selection**
Select all leaf clusters of the condensed cluster tree (clusters with no
children that are also clusters).
- If `cluster_selection_epsilon > 0`, apply the same epsilon search as EOM.
- Dependencies: FR-7.1.

**FR-8.3: Epsilon Search**
Given a set of candidate clusters, a cluster tree, and
`cluster_selection_epsilon`:
- For each candidate whose epsilon < threshold, traverse upward to find the
  ancestor whose epsilon >= threshold.
- Deduplicate: if multiple candidates merge into the same ancestor, keep only
  the ancestor.
- Dependencies: FR-7.1.

**FR-8.4: Labeling and Probabilities**
Given selected clusters:
- Assign each data point a cluster label (0-indexed) or -1 for noise.
- Use union-find on the condensed tree: points not in any selected cluster are
  traced to their root; if root is the tree root and `allow_single_cluster` is
  false, label as noise.
- Compute membership probabilities: `prob(p) = min(lambda(p), max_lambda(C)) / max_lambda(C)`.
- Compute per-cluster stability scores: `stability(C) / (|C| * max_lambda)`.
- `allow_single_cluster` (default false): when true, the root cluster itself
  can be selected as a valid cluster.
- `match_reference_implementation` (default false): when true, adjust
  `min_samples -= 1`, `min_cluster_size += 1`, disable approximate MST; also
  changes labeling logic for edge-case point assignment.
- Dependencies: FR-8.1 or FR-8.2.

---

### FR-9: GLOSH Outlier Scores

**FR-9.1: GLOSH Outlier Detection**
Given a condensed tree, compute an outlier score for each point:
```
outlier_score(p) = (max_lambda(cluster(p)) - lambda(p)) / max_lambda(cluster(p))
```
Where `max_lambda` is propagated upward through the tree so that each cluster's
max_lambda reflects the maximum lambda seen in any of its descendants.
- Score range: [0, 1]. Higher = more outlier-like.
- Points at the maximum lambda of their cluster get score 0.
- Points with infinite lambda get score 0.
- Dependencies: FR-7.1.

---

### FR-10: Top-Level API

**FR-10.1: Cluster Function**
```go
func Cluster(data [][]float64, cfg Config) (*Result, error)
```
- `Config` fields (with defaults):
  - `MinClusterSize int` (default 5, must be >= 2)
  - `MinSamples int` (default: same as MinClusterSize, must be >= 1)
  - `Metric DistanceMetric` (default: Euclidean)
  - `ClusterSelectionMethod string` ("eom" or "leaf", default "eom")
  - `Alpha float64` (default 1.0, must be > 0)
  - `AllowSingleCluster bool` (default false)
  - `ClusterSelectionEpsilon float64` (default 0.0, must be >= 0)
  - `ClusterSelectionPersistence float64` (default 0.0, must be >= 0)
  - `MaxClusterSize int` (default 0 = unlimited)
  - `ClusterSelectionEpsilonMax float64` (default +Inf)
  - `MatchReferenceImplementation bool` (default false)

- `Result` fields:
  - `Labels []int` — cluster label per point (-1 = noise)
  - `Probabilities []float64` — membership strength per point
  - `Stabilities map[int]float64` — stability per selected cluster
  - `OutlierScores []float64` — GLOSH scores per point
  - `CondensedTree` — the condensed tree structure
  - `SingleLinkageTree` — the full dendrogram

**FR-10.2: Input Validation**
- `MinClusterSize` must be >= 2.
- `MinSamples` must be >= 1.
- `Alpha` must be > 0.
- `ClusterSelectionEpsilon` must be >= 0.
- `ClusterSelectionPersistence` must be >= 0.
- `ClusterSelectionMethod` must be "eom" or "leaf".
- `ClusterSelectionEpsilonMax` must be >= `ClusterSelectionEpsilon`.
- When `MinSamples` defaults (is 0 or unset), it takes the value of
  `MinClusterSize`.
- `MinSamples` is clamped to `min(n-1, MinSamples)` at runtime.
- Return clear errors for invalid configurations.

**FR-10.3: Precomputed Distance Matrix API**
Accept a flat `[]float64` or equivalent matrix type as a precomputed distance
matrix. Must be square and symmetric. May contain `+Inf` for missing distances.
- Dependencies: FR-1.7.

---

### FR-11: Approximate Prediction (Phase 3)

**FR-11.1: Prediction Data Cache**
After clustering, optionally cache:
- The spatial tree (KD or ball) built from the original data.
- Core distances for all training points.
- A cluster map (condensed tree cluster ID -> output label).
- Max lambda values per cluster.
- Exemplar points per cluster (points at the leaf max lambda).

**FR-11.2: Approximate Predict**
Given cached prediction data and new unseen points:
- Query the spatial tree for 2*min_samples nearest neighbors.
- Compute mutual reachability distance to the nearest neighbor.
- Walk the condensed tree to find the appropriate cluster.
- Return predicted label and membership probability.
- Dependencies: FR-11.1, FR-3.1 or FR-3.2.

**FR-11.3: Approximate Outlier Scores**
Given cached prediction data and new points, compute GLOSH-style outlier scores.
- Dependencies: FR-11.1.

**FR-11.4: Membership Vectors**
Compute soft membership probability vectors over all clusters for new points,
combining distance-based and outlier-based membership components.
- Dependencies: FR-11.1.

**FR-11.5: All-Points Membership Vectors**
Compute soft membership vectors for all points in the original training data
(more efficient than per-point prediction since points are already in the tree).
- Dependencies: FR-11.1.

---

### FR-12: Robust Single Linkage (Phase 3)

**FR-12.1: Robust Single Linkage Clustering**
Thin wrapper around the core pipeline (core distances -> mutual reachability ->
MST -> single-linkage tree) with a different default alpha (`sqrt(2)`) and a
`cut` parameter to produce flat clusters at a given distance threshold.
- `gamma` parameter: minimum cluster size for flat clustering (default 5).
- Dependencies: FR-4, FR-5, FR-6.

---

### FR-13: Leaf Cluster Selection (Phase 3)

Already covered in FR-8.2.

---

### FR-14: DBCV Cluster Validity Index (Phase 3)

**FR-14.1: Full DBCV Index**
Compute the Density-Based Cluster Validity (DBCV) index for a clustering:
- For each cluster: compute all-points-core-distances, mutual reachability
  distances, internal MST, and density sparseness (max internal MST edge).
- Between clusters: compute density separation (min mutual reachability
  distance between internal MST nodes).
- Per-cluster validity: `(min_separation - sparseness) / max(min_separation, sparseness)`.
- Overall: weighted average by cluster size.
- Range: [-1, 1]. Higher is better.
- Dependencies: FR-1, FR-5.1.

**FR-14.2: Relative Validity (Fast Approximation)**
Compute an approximate DBCV score using the mutual reachability MST
(already computed during clustering) instead of per-cluster all-points MSTs.
- Faster but less precise than FR-14.1.
- Dependencies: FR-5, FR-8.

---

### FR-15: FLASC Branch Detection (Phase 3)

**FR-15.1: Branch Detection Data**
Cache intermediate results needed for detecting branch hierarchies within
clusters (raw data, labels, condensed tree, spatial tree info).
- Dependencies: FR-10.1, FR-3.

**FR-15.2: Branch Detection**
Detect sub-cluster branches within identified clusters.
- Dependencies: FR-15.1.

---

### FR-16: Soft Clustering (Phase 3)

Covered by FR-11.4 and FR-11.5 (membership vectors).

---

## Non-Functional Requirements

### NFR-1: Performance

**NFR-1.1: Phase 1 Complexity**
The brute-force pipeline (dense distance matrix + Prim's MST) must have O(n^2)
time and space complexity. This is the baseline for correctness verification.

**NFR-1.2: Phase 2 Complexity**
With KD-tree or ball tree acceleration plus Boruvka MST, the pipeline must
achieve O(n log n) time complexity for low-dimensional data (d << 60).

**NFR-1.3: Competitive with Python Reference**
Phase 2 performance on datasets of 10k-100k points should be competitive with
(within 2x of) the Python/Cython reference implementation.

**NFR-1.4: Cache-Friendly Data Layout**
Use flat `[]float64` with stride-based indexing (row-major) for the primary data
matrix, not `[][]float64`, to maximize cache locality. `gonum/mat.Dense` is
acceptable where convenient.

**NFR-1.5: Goroutine-Safe Data Structures**
Even if Phase 1 is single-threaded, data structures must be designed for safe
concurrent read access (no hidden mutable shared state during computation).

**NFR-1.6: Parallel Distance Computation (Phase 2)**
Support parallel pairwise distance computation and k-NN queries via goroutine
worker pools. Number of workers should be configurable.

---

### NFR-2: Correctness

**NFR-2.1: Golden Test Suite**
Generate ground-truth outputs from the Python reference implementation on known
datasets (e.g., scikit-learn's `make_blobs`, `make_moons`, the bundled
`clusterable_data.npy`). Go tests must match these outputs:
- Labels must match exactly (accounting for arbitrary label ordering).
- Probabilities must match within float64 epsilon (1e-10).
- Stabilities must match within float64 epsilon.
- Outlier scores must match within float64 epsilon.

**NFR-2.2: Edge Cases**
Must handle and test:
- Single-point input (n=1).
- Two-point input (n=2).
- All points identical (zero distances).
- `min_cluster_size` > n (all noise).
- `min_samples` > n (clamped to n-1).
- Datasets where all points are noise.
- Datasets producing a single cluster (with `allow_single_cluster`).
- Distance matrices with +Inf entries.

**NFR-2.3: Numerical Stability**
- Lambda values computed as 1/distance must handle distance=0 (lambda=+Inf).
- Probability normalization must handle zero-sum edge cases.
- Stability computation must not overflow for large clusters with high lambda.

---

### NFR-3: API Design

**NFR-3.1: Go-Idiomatic API**
- Functional API with options pattern, not OOP fit/predict.
- Exported types use Go naming conventions (PascalCase).
- Errors returned as `error` values, not panics.
- No scikit-learn API compatibility required.

**NFR-3.2: gonum Interoperability**
Accept `gonum/mat.Dense` as input where practical. Internal computations may use
`gonum` for matrix operations and linear algebra.

**NFR-3.3: Zero Dependencies Beyond stdlib + gonum**
The core library should depend only on the Go standard library and `gonum`.
No CGO dependencies.

---

### NFR-4: Code Quality

**NFR-4.1: Test Coverage**
- Every pipeline stage must have independent unit tests.
- Integration tests exercising the full pipeline on reference datasets.
- Benchmark tests for performance-critical functions (distance computation, MST,
  condensed tree operations).

**NFR-4.2: Documentation**
- Package-level doc comment explaining the algorithm and usage.
- Exported functions and types must have godoc comments.
- No excessive internal comments — code should be self-explanatory.

---

## Requirement Dependencies

```
FR-1 (Distance Metrics)
  └──> FR-2 (Core Distances)
         └──> FR-4 (Mutual Reachability)
                └──> FR-5 (MST)
                       └──> FR-6 (Single-Linkage / Union-Find)
                              └──> FR-7 (Condensed Tree)
                                     ├──> FR-8 (Cluster Extraction)
                                     │      └──> FR-10 (Top-Level API)
                                     └──> FR-9 (GLOSH Outlier Scores)
                                            └──> FR-10 (Top-Level API)

FR-3 (Spatial Trees)         [Phase 2]
  ├──> FR-2.2, FR-2.3        [Tree-accelerated core distances]
  ├──> FR-5.2, FR-5.3        [Prim's vector / Boruvka MST]
  └──> FR-11 (Prediction)    [Phase 3]

FR-11 (Prediction)           [Phase 3, depends on FR-3 + FR-7 + FR-8]
FR-12 (Robust Single Link)   [Phase 3, depends on FR-4 + FR-5 + FR-6]
FR-14 (DBCV Validity)        [Phase 3, depends on FR-1 + FR-5]
FR-15 (FLASC Branches)       [Phase 3, depends on FR-3 + FR-10]
```

### MVP Implementation Order (Phase 1)

1. FR-1.1–FR-1.6 — Distance metrics + interface
2. FR-2.1 — Brute-force core distances
3. FR-4.1 — Dense mutual reachability
4. FR-5.1 — Prim's MST on dense matrix
5. FR-6.1–FR-6.2 — Union-Find + label function
6. FR-7.1–FR-7.2 — Condensed tree + stability
7. FR-8.1, FR-8.4 — EOM extraction + labeling
8. FR-9.1 — GLOSH outlier scores
9. FR-10.1–FR-10.2 — Top-level API + validation
10. FR-8.2–FR-8.3 — Leaf selection + epsilon search
11. FR-1.7, FR-10.3 — Precomputed distance matrix support
12. FR-7.3 — Persistence-based simplification
