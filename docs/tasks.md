# Implementation Tasks: hdbscan-go

Tasks broken down from `docs/requirements.md` for Phase 1 (MVP) implementation.
Phase 2 and Phase 3 tasks are listed separately at the end.

---

## Phase 1: MVP — Brute-Force HDBSCAN Pipeline

### Task 1: Project Scaffolding & Go Module Setup

**Description:** Initialize the Go module, directory structure, and build tooling.

**Implementation approach:**
- `go mod init` with module path (e.g., `github.com/TrevorS/hdbscan`)
- Add `gonum` dependency (`gonum.org/v1/gonum`)
- Create package structure: single `hdbscan` package for Phase 1 (flat, no sub-packages)
- Add Makefile with `test`, `bench`, `lint`, `fmt` targets
- Add `.golangci-lint.yml` with strict settings

**Required components:**
- `go.mod`, `go.sum`
- `Makefile`
- `.golangci-lint.yml`

**Acceptance criteria:**
- `go build ./...` succeeds
- `go test ./...` runs (even with no tests yet)
- `make lint` runs golangci-lint

**Dependencies:** None

---

### Task 2: Distance Metric Interface & Built-in Metrics (FR-1.1–FR-1.6)

**Description:** Define the `DistanceMetric` interface and `DistanceFunc` type. Implement Euclidean, Manhattan, Cosine, Chebyshev, and Minkowski metrics.

**Implementation approach (TDD):**
1. Write tests first: for each metric, test against known input/output pairs (hand-computed and cross-checked against `scipy.spatial.distance`)
2. Define types:
   ```go
   type DistanceFunc func(a, b []float64) float64

   type DistanceMetric interface {
       Distance(a, b []float64) float64
       ReducedDistance(a, b []float64) float64
   }
   ```
3. Implement adapter: `DistanceFunc` → `DistanceMetric` (where `ReducedDistance` == `Distance`)
4. Implement each metric as a struct satisfying `DistanceMetric`:
   - `EuclideanMetric` — `ReducedDistance` returns squared Euclidean (no sqrt)
   - `ManhattanMetric` — `ReducedDistance` == `Distance`
   - `CosineMetric` — 1 - cosine_similarity; `ReducedDistance` == `Distance`
   - `ChebyshevMetric` — max(|a[i]-b[i]|)
   - `MinkowskiMetric` — parameterized by `P float64`; reject P < 0; verify P=1→Manhattan, P=2→Euclidean equivalence

**Required components:**
- `distance.go` — types + implementations
- `distance_test.go` — unit tests per metric

**Test requirements:**
- Each metric tested on: zero vectors, unit vectors, identical vectors (distance=0), known hand-computed values
- Minkowski equivalence: P=1 matches Manhattan, P=2 matches Euclidean
- Minkowski rejects negative P
- DistanceFunc adapter produces correct DistanceMetric behavior

**Acceptance criteria:**
- All built-in metrics pass unit tests
- Custom `DistanceFunc` works through the adapter
- Minkowski special cases verified

**Dependencies:** Task 1

---

### Task 3: Brute-Force Core Distance Computation (FR-2.1)

**Description:** Given a full pairwise distance matrix and `minSamples`, compute the core distance for each point as the distance to its k-th nearest neighbor.

**Implementation approach (TDD):**
1. Write tests first: small hand-computed examples (3–5 points), verify core distances
2. Implement `ComputeCoreDistances(distMatrix []float64, n, minSamples int) []float64`
   - Use flat `[]float64` row-major for the distance matrix (NFR-1.4)
   - For each row, find the k-th smallest value using `nth_element` selection (partial sort via `container/heap` or quickselect)
   - Clamp `minSamples` to `min(n-1, minSamples)`
3. Also implement `ComputePairwiseDistances(data []float64, rows, cols int, metric DistanceMetric) []float64`
   - Takes flat row-major data, returns flat n×n distance matrix

**Required components:**
- `core_distance.go` — core distance + pairwise distance computation
- `core_distance_test.go`

**Test requirements:**
- 3-point Euclidean example with known core distances
- minSamples=1 returns nearest-neighbor distance
- minSamples >= n-1 returns max distance in row
- minSamples > n clamped to n-1

**Acceptance criteria:**
- Core distances match hand-computed values
- Partial sort used (not full sort) for efficiency
- Flat `[]float64` layout with stride-based indexing

**Dependencies:** Task 2

---

### Task 4: Dense Mutual Reachability Distance (FR-4.1)

**Description:** Compute the mutual reachability distance matrix from a pairwise distance matrix, core distances, and alpha parameter.

**Implementation approach (TDD):**
1. Write tests: small examples where MR distances can be hand-verified
2. Implement `MutualReachability(distMatrix, coreDistances []float64, n int, alpha float64) []float64`
   - For each pair (i,j): `mr[i][j] = max(coreDistances[i], coreDistances[j], distMatrix[i*n+j] / alpha)`
   - When `alpha == 1.0`, skip the division (optimization + avoids float imprecision)
   - Output is a flat n×n `[]float64` (symmetric)
   - Diagonal is 0 (or core distance — match reference behavior)

**Required components:**
- `reachability.go`
- `reachability_test.go`

**Test requirements:**
- 3-point example: verify each element of MR matrix
- Alpha=1.0 fast path produces same result as alpha=1.0 through division path
- Alpha=0.5 doubles effective distances before max comparison
- Verify symmetry of output

**Acceptance criteria:**
- Element-wise match with hand-computed MR distances
- Alpha=1.0 optimization path verified

**Dependencies:** Task 3

---

### Task 5: Prim's MST on Dense Matrix (FR-5.1)

**Description:** Implement Prim's algorithm to compute a minimum spanning tree from a dense mutual reachability distance matrix.

**Implementation approach (TDD):**
1. Write tests: small graphs (4–6 points) with known MSTs
2. Implement `PrimMST(mrMatrix []float64, n int) [][3]float64`
   - Output: (n-1) edges as `[from, to, weight]`
   - Use the "lazy" Prim's with a flat array tracking minimum edge to tree (matches reference `mst_linkage_core`)
   - No priority queue needed for dense matrix — just scan the `current_distances` array each iteration
   - Track `in_tree []bool`, `current_distances []float64`, `current_sources []int`
   - Each iteration: find min `current_distances[j]` where `!in_tree[j]`, add that edge, update distances
3. Warn (log) if any MST edge weight is +Inf

**Required components:**
- `mst.go`
- `mst_test.go`

**Test requirements:**
- 4-point complete graph with known MST
- Graph with +Inf edges (warn but succeed)
- Single point input (n=1): returns empty edge list
- Two points: returns single edge

**Acceptance criteria:**
- MST weight matches known optimal
- O(n²) performance (n iterations × n scan each)
- +Inf edges produce warning, not error

**Dependencies:** Task 4

---

### Task 6: Union-Find & Single-Linkage Label Function (FR-6.1, FR-6.2)

**Description:** Implement a union-find data structure and the `label` function that converts sorted MST edges into a single-linkage dendrogram.

**Implementation approach (TDD):**
1. Write tests: union-find operations, then full label function on small MSTs
2. Implement `UnionFind` struct:
   ```go
   type UnionFind struct {
       parent []int
       size   []int
   }
   ```
   - `Find(x int) int` — with path compression
   - `Union(x, y int)` — union by size
3. Implement `Label(mstEdges [][3]float64, n int) [][4]float64`
   - Sort edges by weight (ascending)
   - Process each edge: find roots of both endpoints, union them
   - Output row: `[left, right, distance, mergedSize]`
   - New cluster IDs start at `n` and increment
   - Output is (n-1) × 4

**Required components:**
- `unionfind.go` + `unionfind_test.go`
- `label.go` + `label_test.go`

**Test requirements:**
- UnionFind: basic union/find, path compression verified, size tracking
- Label function: 4-point MST → known dendrogram
- Edge case: n=1 (empty dendrogram), n=2 (single row)

**Acceptance criteria:**
- Dendrogram matches scipy-format single-linkage output for test cases
- Cluster IDs start at n and increment correctly
- Merged sizes are correct

**Dependencies:** Task 5

---

### Task 7: Condensed Tree Construction (FR-7.1)

**Description:** Walk a single-linkage dendrogram and produce a condensed tree by collapsing clusters smaller than `minClusterSize`.

**Implementation approach (TDD):**
1. Write tests: build dendrogram from known data, verify condensed tree output
2. Define condensed tree record:
   ```go
   type CondensedTreeEntry struct {
       Parent    int
       Child     int
       LambdaVal float64
       ChildSize int
   }
   ```
3. Implement `CondenseTree(dendrogram [][4]float64, minClusterSize int) []CondensedTreeEntry`
   - Walk dendrogram in BFS order from root (last row = root)
   - Lambda = 1.0 / distance (if distance == 0, lambda = +Inf)
   - Logic per node:
     - If both children have size >= minClusterSize → create two new cluster children
     - If both children have size < minClusterSize → collapse all points into parent
     - If one child < minClusterSize → keep larger as same cluster, collapse smaller
   - Use a queue/stack for BFS traversal
   - Track relabeling: dendrogram node IDs → condensed tree cluster IDs

**Required components:**
- `condensed_tree.go`
- `condensed_tree_test.go`

**Test requirements:**
- 6-point dataset: verify condensed tree entries match expected structure
- minClusterSize=2 vs minClusterSize=3 produce different trees
- All-identical points (distance=0 → lambda=+Inf): handled without panic
- Single cluster scenario

**Acceptance criteria:**
- Condensed tree entries match reference implementation output for test data
- Lambda computation handles zero distances
- BFS traversal order matches reference

**Dependencies:** Task 6

---

### Task 8: Stability Computation (FR-7.2)

**Description:** Compute stability scores for each cluster in the condensed tree.

**Implementation approach (TDD):**
1. Write tests: small condensed tree with known stability values
2. Implement `ComputeStability(tree []CondensedTreeEntry) map[int]float64`
   - For each cluster: find its `lambdaBirth` (the lambda at which it first appears as a child)
   - Root cluster has `lambdaBirth = 0`
   - Stability = sum over all point entries in cluster: `(lambdaVal - lambdaBirth) * childSize`
   - Only sum entries where `childSize == 1` (individual points) — this is the standard definition
   - Return map from cluster ID to stability

**Required components:**
- `stability.go`
- `stability_test.go`

**Test requirements:**
- Hand-computed stability for a small condensed tree
- Root cluster with lambdaBirth=0
- Cluster with all points at same lambda (stability reflects uniform density)

**Acceptance criteria:**
- Stability values match hand-computed expected values
- Root cluster lambdaBirth is 0
- All clusters in the condensed tree have a stability entry

**Dependencies:** Task 7

---

### Task 9: EOM Cluster Selection (FR-8.1)

**Description:** Implement the Excess-of-Mass cluster selection method that walks the condensed tree bottom-up, selecting clusters that maximize total stability.

**Implementation approach (TDD):**
1. Write tests: condensed trees with known optimal cluster selections
2. Implement `SelectClustersEOM(tree []CondensedTreeEntry, stability map[int]float64, maxClusterSize int, clusterSelectionEpsilonMax float64) (selectedClusters map[int]bool, updatedStability map[int]float64)`
   - Build parent→children map from condensed tree (cluster nodes only, not points)
   - Walk bottom-up (leaves first):
     - For each non-leaf cluster: compare own stability vs sum of children's stabilities
     - If children's total > own stability (or own size exceeds `maxClusterSize`, or own epsilon > `clusterSelectionEpsilonMax`): mark as NOT selected, set stability = children's sum
     - Else: mark as selected, mark all descendants as NOT selected
   - Handle `clusterSelectionEpsilon > 0` via epsilon search (Task 11)

**Required components:**
- `cluster_selection.go`
- `cluster_selection_test.go`

**Test requirements:**
- Simple 2-cluster case: both selected
- Nested clusters: parent vs children stability comparison
- maxClusterSize forcing split
- Single cluster (allow_single_cluster=false → noise, =true → one cluster)

**Acceptance criteria:**
- Selected clusters match expected for test cases
- Stability propagation (children sum replaces parent) is correct
- maxClusterSize constraint works

**Dependencies:** Task 8

---

### Task 10: Labeling & Probabilities (FR-8.4)

**Description:** Given selected clusters, assign each point a cluster label and compute membership probabilities.

**Implementation approach (TDD):**
1. Write tests: known selected clusters → expected labels and probabilities
2. Implement `GetLabelsAndProbabilities(tree []CondensedTreeEntry, selectedClusters map[int]bool, n int, allowSingleCluster bool) (labels []int, probabilities []float64)`
   - Use union-find on condensed tree to assign each point to its nearest selected ancestor cluster
   - Points not in any selected cluster: label = -1 (noise)
   - If `allowSingleCluster` is false and point traces to root: label = -1
   - Compute probabilities: `prob(p) = min(lambda(p), maxLambda(C)) / maxLambda(C)`
   - maxLambda(C) = maximum lambda of any point in cluster C
3. Implement `ComputeClusterPersistence(tree []CondensedTreeEntry, selectedClusters map[int]bool, stability map[int]float64) map[int]float64`
   - Per-cluster score: `stability(C) / (|C| * maxLambda(C))`

**Required components:**
- `labeling.go`
- `labeling_test.go`

**Test requirements:**
- Simple case: 2 clusters, correct labels assigned
- Noise points labeled -1
- allowSingleCluster=true: root cluster is valid
- allowSingleCluster=false: unattached points are noise
- Probabilities in [0, 1] range

**Acceptance criteria:**
- Labels and probabilities match reference for test data
- Noise handling correct for both allowSingleCluster modes

**Dependencies:** Task 9

---

### Task 11: Epsilon Search & Leaf Cluster Selection (FR-8.2, FR-8.3)

**Description:** Implement epsilon-based cluster merging and the leaf cluster selection method.

**Implementation approach (TDD):**
1. Write tests: epsilon search on known tree, leaf selection on known tree
2. Implement `EpsilonSearch(tree []CondensedTreeEntry, candidateClusters map[int]bool, clusterSelectionEpsilon float64) map[int]bool`
   - For each candidate: epsilon = 1/lambdaBirth
   - If epsilon < threshold: traverse upward to find ancestor with epsilon >= threshold
   - Deduplicate: if multiple candidates merge to same ancestor, keep only ancestor
3. Implement `SelectClustersLeaf(tree []CondensedTreeEntry, clusterSelectionEpsilon float64) map[int]bool`
   - Find all leaf clusters (clusters with no cluster children)
   - If clusterSelectionEpsilon > 0: apply epsilon search

**Required components:**
- `epsilon_search.go` (or add to `cluster_selection.go`)
- Tests in `cluster_selection_test.go`

**Test requirements:**
- Epsilon search: candidate below threshold merges upward
- Epsilon search: deduplication when siblings merge to same parent
- Leaf selection: correct leaves identified
- Leaf selection with epsilon: leaves merged upward correctly

**Acceptance criteria:**
- Epsilon search produces correct merged cluster set
- Leaf selection identifies correct leaf clusters
- Integration with EOM (epsilon applied after EOM selection)

**Dependencies:** Task 9

---

### Task 12: GLOSH Outlier Scores (FR-9.1)

**Description:** Compute GLOSH outlier scores for each point from the condensed tree.

**Implementation approach (TDD):**
1. Write tests: small condensed tree with known outlier scores
2. Implement `OutlierScores(tree []CondensedTreeEntry) []float64`
   - For each cluster: compute maxLambda by propagating upward through the tree
     - maxLambda of a cluster = max of all lambda values of its point members, and maxLambda of its child clusters
   - For each point: `score = (maxLambda(cluster) - lambda(point)) / maxLambda(cluster)`
   - If maxLambda == 0: score = 0
   - If lambda(point) == +Inf: score = 0
   - Score range: [0, 1]

**Required components:**
- `outlier.go`
- `outlier_test.go`

**Test requirements:**
- Point at cluster's maxLambda → score 0
- Point departing early → score near 1
- Cluster with maxLambda=0 (infinite distance) → score 0
- All scores in [0, 1]

**Acceptance criteria:**
- Outlier scores match reference implementation for test data
- Edge cases handled (zero distance, infinite lambda)

**Dependencies:** Task 7

---

### Task 13: Top-Level API & Config Validation (FR-10.1, FR-10.2)

**Description:** Implement the `Cluster` function, `Config` struct with defaults, and `Result` struct. Wire together the full pipeline.

**Implementation approach (TDD):**
1. Write validation tests first (invalid configs → errors)
2. Define types:
   ```go
   type Config struct {
       MinClusterSize              int
       MinSamples                  int
       Metric                      DistanceMetric
       ClusterSelectionMethod      string
       Alpha                       float64
       AllowSingleCluster          bool
       ClusterSelectionEpsilon     float64
       ClusterSelectionPersistence float64
       MaxClusterSize              int
       ClusterSelectionEpsilonMax  float64
       MatchReferenceImplementation bool
   }

   type Result struct {
       Labels           []int
       Probabilities    []float64
       Stabilities      map[int]float64
       OutlierScores    []float64
       CondensedTree    []CondensedTreeEntry
       SingleLinkageTree [][4]float64
   }
   ```
3. Implement `Cluster(data [][]float64, cfg Config) (*Result, error)`:
   - Apply defaults: MinClusterSize=5, MinSamples=MinClusterSize if unset, Alpha=1.0, Method="eom", EpsilonMax=+Inf
   - Validate config (return errors for invalid values)
   - Convert `[][]float64` to flat representation
   - Pipeline: distances → core distances → mutual reachability → MST → label → condense → stability → select → label+probs → outliers
   - Apply persistence simplification if threshold > 0
   - Apply epsilon search if epsilon > 0
   - Handle `MatchReferenceImplementation` flag
4. Also implement convenience: `DefaultConfig() Config`

**Required components:**
- `hdbscan.go` — Config, Result, Cluster function, DefaultConfig
- `hdbscan_test.go` — validation tests, integration tests

**Test requirements:**
- Config validation: MinClusterSize < 2 → error, MinSamples < 1 → error, Alpha <= 0 → error, invalid method → error
- Default config applied correctly
- MinSamples defaults to MinClusterSize when unset
- MinSamples clamped to n-1 at runtime

**Acceptance criteria:**
- `Cluster()` produces correct results on small hand-computed datasets
- All invalid configs return descriptive errors
- Defaults match specification

**Dependencies:** Tasks 2–12

---

### Task 14: Precomputed Distance Matrix Support (FR-1.7, FR-10.3)

**Description:** Accept a precomputed distance matrix as input instead of raw data.

**Implementation approach (TDD):**
1. Write tests: pass precomputed matrix, verify same results as computing from data
2. Add `ClusterPrecomputed(distMatrix []float64, n int, cfg Config) (*Result, error)`
   - Validate matrix is square (n×n)
   - Allow `+Inf` values (missing distances)
   - Skip pairwise distance computation, go directly to core distances → pipeline
3. Alternatively, add a `Precomputed bool` field to Config and accept data as a flat matrix

**Required components:**
- Add to `hdbscan.go`
- Tests in `hdbscan_test.go`

**Test requirements:**
- Precomputed matrix produces same result as computing from raw data
- Non-square matrix → error
- +Inf values handled (warn on MST edge)

**Acceptance criteria:**
- Precomputed path produces identical results to raw-data path for same dataset
- Validation rejects non-square input

**Dependencies:** Task 13

---

### Task 15: Persistence-Based Simplification (FR-7.3)

**Description:** Remove low-persistence leaf clusters from the condensed tree before cluster extraction.

**Implementation approach (TDD):**
1. Write tests: condensed tree with known low-persistence leaves
2. Implement `SimplifyHierarchy(tree []CondensedTreeEntry, persistenceThreshold float64) []CondensedTreeEntry`
   - For each leaf cluster: compute persistence = lambdaBirth(cluster) - lambdaBirth(parent)
   - If persistence < threshold: remove the leaf cluster, re-parent its points to the parent
   - Relabel remaining clusters for consecutive numbering

**Required components:**
- `simplify.go` (or add to `condensed_tree.go`)
- `simplify_test.go`

**Test requirements:**
- Low-persistence leaf removed, high-persistence leaf kept
- Relabeling produces consecutive cluster IDs
- Threshold=0 returns tree unchanged

**Acceptance criteria:**
- Simplified tree matches reference behavior
- Point re-parenting correct

**Dependencies:** Task 7

---

### Task 16: Golden Test Suite Generation (NFR-2.1)

**Description:** Generate ground-truth outputs from the Python reference implementation on known datasets, then write Go tests that validate against them.

**Implementation approach:**
1. Write a Python script that runs the reference HDBSCAN on standard datasets:
   - `make_blobs(n_samples=100, centers=3)` with fixed random_state
   - `make_moons(n_samples=200)` with fixed random_state
   - Small hand-crafted datasets (5–10 points)
   - Various config combos: different minClusterSize, EOM vs leaf, allow_single_cluster, alpha values
2. Export for each run: input data, config, labels, probabilities, stabilities, outlier scores, condensed tree, single-linkage tree — as JSON files
3. Write Go tests that load JSON golden files and compare output:
   - Labels: exact match (with label permutation handling)
   - Probabilities: within 1e-10
   - Stabilities: within 1e-10
   - Outlier scores: within 1e-10

**Required components:**
- `testdata/generate_golden.py` — Python script to generate ground truth
- `testdata/*.json` — golden test data files
- `golden_test.go` — Go tests that load and compare

**Test requirements:**
- At least 5 distinct dataset/config combinations
- Label permutation-invariant comparison
- Floating-point tolerance comparison

**Acceptance criteria:**
- All golden tests pass
- Tests cover EOM + leaf methods, various minClusterSize, alpha, allowSingleCluster
- Reproducible: Python script can regenerate identical golden files

**Dependencies:** Task 13

---

### Task 17: Edge Case Tests (NFR-2.2)

**Description:** Test all specified edge cases to ensure robustness.

**Implementation approach (TDD):**
- Write tests for each edge case:
  - n=1 (single point): labels=[noise or single cluster], no crash
  - n=2: either one cluster or two noise points depending on minClusterSize
  - All identical points (zero distances): lambda=+Inf handled
  - minClusterSize > n: all points labeled noise
  - minSamples > n: clamped, no error
  - All noise result: labels all -1, probabilities all 0
  - Single cluster with allowSingleCluster=true
  - Distance matrix with +Inf entries

**Required components:**
- `edge_cases_test.go`

**Test requirements:**
- Each edge case has an explicit test function
- No panics on any edge case
- Results are sensible (not garbage values)

**Acceptance criteria:**
- All edge case tests pass
- No panics, no infinite loops, no NaN in output

**Dependencies:** Task 13

---

### Task 18: Benchmark Tests (NFR-4.1)

**Description:** Write benchmark tests for performance-critical pipeline stages.

**Implementation approach:**
- Benchmarks for:
  - Pairwise distance computation (100, 500, 1000 points)
  - Core distance computation
  - Mutual reachability
  - Prim's MST
  - Condense tree
  - Full pipeline end-to-end
- Use `testing.B` with `b.ResetTimer()` after setup

**Required components:**
- `benchmark_test.go`

**Acceptance criteria:**
- Benchmarks run via `go test -bench=.`
- O(n²) scaling observable in results
- No unexpected allocations in hot paths (`b.ReportAllocs()`)

**Dependencies:** Task 13

---

## Phase 2: Performance Optimization (Future)

These tasks depend on the complete Phase 1 pipeline.

### Task P2-1: KD-Tree with k-NN Query (FR-3.1)
Build a KD-tree supporting k-nearest-neighbor queries. Configurable leaf size (default 40). Valid for Euclidean, Manhattan, Chebyshev, Minkowski metrics.

**Dependencies:** Phase 1 complete

### Task P2-2: Ball Tree with k-NN Query (FR-3.2)
Build a ball tree supporting k-NN queries for all metric types. Configurable leaf size.

**Dependencies:** Phase 1 complete

### Task P2-3: Tree-Accelerated Core Distances (FR-2.2, FR-2.3)
Use KD-tree or ball tree k-NN queries to compute core distances without the full distance matrix.

**Dependencies:** P2-1 or P2-2

### Task P2-4: Prim's MST with Core Distance Vector (FR-5.2)
Implement `mst_linkage_core_vector`: Prim's MST computing mutual reachability on-the-fly from raw data + core distances + metric, avoiding the full n×n matrix.

**Dependencies:** P2-3

### Task P2-5: Dual-Tree Traversal (FR-3.3)
Support dual-tree query patterns for the Boruvka MST algorithm.

**Dependencies:** P2-1 or P2-2

### Task P2-6: Boruvka Dual-Tree MST (FR-5.3)
Implement the dual-tree Boruvka MST construction for O(n log n) performance on low-dimensional data.

**Dependencies:** P2-5

### Task P2-7: Parallel Distance Computation (NFR-1.6)
Add goroutine worker pool for parallel pairwise distance computation and k-NN queries.

**Dependencies:** P2-3

---

## Phase 3: Full Feature Parity (Future)

### Task P3-1: Additional Distance Metrics (FR-1.8)
Haversine, Mahalanobis, Hamming, Canberra, Bray-Curtis, Jaccard, Dice, Standardized Euclidean, Weighted Minkowski, Rogers-Tanimoto, Russell-Rao, Sokal-Michener, Sokal-Sneath.

### Task P3-2: Prediction Data Cache (FR-11.1)
Cache spatial tree, core distances, cluster map, max lambdas, and exemplar points after clustering.

### Task P3-3: Approximate Predict (FR-11.2)
Predict cluster membership for new unseen points using cached prediction data.

### Task P3-4: Approximate Outlier Scores for New Points (FR-11.3)
Compute GLOSH-style outlier scores for new points.

### Task P3-5: Membership Vectors (FR-11.4, FR-11.5)
Soft membership probability vectors over all clusters, for both new points and training data.

### Task P3-6: Robust Single Linkage (FR-12.1)
Thin wrapper with different default alpha (sqrt(2)) and flat cut parameter.

### Task P3-7: DBCV Cluster Validity Index (FR-14.1, FR-14.2)
Full and approximate DBCV scoring for evaluating clustering quality.

### Task P3-8: FLASC Branch Detection (FR-15.1, FR-15.2)
Detect sub-cluster branches within identified clusters.

---

## Task Dependency Graph (Phase 1)

```
Task 1 (Scaffolding)
  └──> Task 2 (Distance Metrics)
         └──> Task 3 (Core Distances)
                └──> Task 4 (Mutual Reachability)
                       └──> Task 5 (Prim's MST)
                              └──> Task 6 (Union-Find + Label)
                                     └──> Task 7 (Condensed Tree)
                                            ├──> Task 8 (Stability)
                                            │      └──> Task 9 (EOM Selection)
                                            │             ├──> Task 10 (Labels + Probs)
                                            │             └──> Task 11 (Epsilon + Leaf)
                                            ├──> Task 12 (GLOSH Outliers)
                                            └──> Task 15 (Persistence Simplification)

Tasks 10, 11, 12, 15 ──> Task 13 (Top-Level API)
                            ├──> Task 14 (Precomputed Matrix)
                            ├──> Task 16 (Golden Tests)
                            ├──> Task 17 (Edge Case Tests)
                            └──> Task 18 (Benchmarks)
```
