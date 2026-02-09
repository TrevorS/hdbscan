# Agent Team Prompt: Implement hdbscan-go Phase 1

You are orchestrating a team of agents to implement a Go port of HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise). The full specification lives in the project docs — read them before doing anything.

## Required Reading (Do This First)

Every agent on this team must read these files before writing any code:

- `CLAUDE.md` — architecture, data structures, design decisions, build commands
- `docs/requirements.md` — formal requirements (FR/NFR numbering)
- `docs/tasks.md` — task breakdown with dependencies and acceptance criteria
- `docs/initial-convo.md` — module-by-module analysis of the Python reference
- `docs/brief.md` — project scope and phasing

The reference Python/Cython implementation is at `reference/scikit-learn-contrib-hdbscan/hdbscan/`. Consult it for algorithmic details, but **translate the algorithm, not the Python idioms**. Write idiomatic Go.

## Project State

- **No Go code exists yet.** The repo has docs, CLAUDE.md, and the reference submodule. Everything must be built from scratch.
- **Phase 1 only.** Do not implement Phase 2 (spatial trees, Boruvka) or Phase 3 (prediction, FLASC, etc.) features.
- **Single `hdbscan` package.** No sub-packages. All `.go` files live at the repo root.

## Team Structure

Spawn the following agents. Each agent owns a vertical slice of the pipeline and is responsible for both implementation and tests.

### Agent 1: `scaffolding` (general-purpose)

**Tasks:** 1, 16 (partial)

Set up the Go module and project infrastructure:

1. `go mod init github.com/TrevorS/hdbscan`
2. Add `gonum` dependency: `go get gonum.org/v1/gonum`
3. Create `Makefile` with targets: `test`, `bench`, `lint`, `fmt`, `build`
4. Create `.golangci-lint.yml` with strict settings (govet, errcheck, staticcheck, unused, gosimple, ineffassign)
5. Write `testdata/generate_golden.py` — Python script that runs the reference HDBSCAN on standard datasets and exports golden test data as JSON. Datasets: `make_blobs(n=100, centers=3, random_state=42)`, `make_moons(n=200, random_state=42)`, a small 6-point hand-crafted dataset. Export for each: input data, config, labels, probabilities, stabilities, outlier scores, condensed tree, single-linkage tree. Use multiple config combos (minClusterSize=5/10, EOM/leaf, allowSingleCluster true/false).
6. Run the Python script to generate `testdata/*.json` golden files.
7. Verify: `go build ./...` and `go test ./...` both succeed (even with no test files yet).

**Acceptance:** `go build ./...` succeeds. `make lint` runs. Golden test data exists in `testdata/`.

### Agent 2: `distance-core` (general-purpose)

**Tasks:** 2, 3, 4

Implement the foundation: distance metrics, pairwise distance computation, core distances, and mutual reachability. **Write tests first (TDD).**

**Files to create:**
- `distance.go` + `distance_test.go` — DistanceMetric interface, DistanceFunc adapter, EuclideanMetric (with squared ReducedDistance), ManhattanMetric, CosineMetric, ChebyshevMetric, MinkowskiMetric(P)
- `core_distance.go` + `core_distance_test.go` — `ComputePairwiseDistances(data []float64, n, dims int, metric DistanceMetric) []float64` and `ComputeCoreDistances(distMatrix []float64, n, minSamples int) []float64`
- `reachability.go` + `reachability_test.go` — `MutualReachability(distMatrix, coreDistances []float64, n int, alpha float64) []float64`

**Critical design constraints:**
- Distance/reachability matrices are flat `[]float64`, length n×n, row-major. Access: `matrix[i*n + j]`.
- Input data to `ComputePairwiseDistances` is also flat `[]float64` with stride-based indexing (row = point, col = dimension).
- Core distance uses partial sort (quickselect or nth-element), not full sort.
- Mutual reachability: `mr[i*n+j] = max(coreDist[i], coreDist[j], dist[i*n+j] / alpha)`. Skip division when alpha == 1.0.
- MinkowskiMetric must reject negative P. P=1 must equal Manhattan, P=2 must equal Euclidean.

**Test requirements:**
- Each metric: zero vectors, unit vectors, identical vectors (dist=0), hand-computed values
- Core distances: 3-5 point examples, minSamples=1, minSamples >= n-1, minSamples > n (clamped)
- Mutual reachability: element-wise verification, alpha=1.0 vs alpha!=1.0, symmetry check

**Consult reference files:**
- `reference/scikit-learn-contrib-hdbscan/hdbscan/dist_metrics.pyx` for metric implementations
- `reference/scikit-learn-contrib-hdbscan/hdbscan/_hdbscan_reachability.pyx` for mutual reachability logic

### Agent 3: `mst-linkage` (general-purpose)

**Tasks:** 5, 6

Implement Prim's MST and the union-find + label function that produces the single-linkage dendrogram. **Write tests first (TDD).**

**Files to create:**
- `mst.go` + `mst_test.go` — `PrimMST(mrMatrix []float64, n int) [][3]float64`
- `unionfind.go` + `unionfind_test.go` — UnionFind struct with Find (path compression) and Union (by size)
- `label.go` + `label_test.go` — `Label(mstEdges [][3]float64, n int) [][4]float64`

**Critical design constraints:**
- Prim's MST: no priority queue needed for dense matrix. Use flat arrays: `inTree []bool`, `currentDistances []float64`, `currentSources []int`. Each iteration: scan for min distance not in tree, add edge, update neighbors. This matches the reference `mst_linkage_core` approach.
- MST output: (n-1) edges as `[from, to, weight]` (float64 for consistency with dendrogram format).
- Warn (log.Printf) if any MST edge weight is +Inf. Do not error.
- Label function: sort MST edges by weight ascending, process with union-find. Output: (n-1) x 4 dendrogram rows `[left, right, distance, mergedSize]`. New cluster IDs start at n and increment.
- UnionFind needs `parent []int`, `size []int`. Path compression in Find, union by size.

**Test requirements:**
- MST: 4-6 point complete graph with known MST weight, +Inf edges, n=1 (empty), n=2 (single edge)
- UnionFind: basic ops, path compression, size tracking
- Label: 4-point MST → known scipy-format dendrogram, n=1, n=2

**Consult reference files:**
- `reference/scikit-learn-contrib-hdbscan/hdbscan/_hdbscan_linkage.pyx` — `mst_linkage_core()` and `label()` functions

### Agent 4: `tree-extraction` (general-purpose)

**Tasks:** 7, 8, 9, 10, 11, 12, 15

Implement the condensed tree, stability, cluster selection (EOM + leaf), labeling/probabilities, GLOSH outlier scores, and persistence simplification. This is the algorithmic heart. **Write tests first (TDD).**

**Files to create:**
- `condensed_tree.go` + `condensed_tree_test.go` — CondensedTreeEntry struct, `CondenseTree(dendrogram [][4]float64, minClusterSize int) []CondensedTreeEntry`
- `stability.go` + `stability_test.go` — `ComputeStability(tree []CondensedTreeEntry) map[int]float64`
- `cluster_selection.go` + `cluster_selection_test.go` — `SelectClustersEOM(...)`, `SelectClustersLeaf(...)`, `EpsilonSearch(...)`
- `labeling.go` + `labeling_test.go` — `GetLabelsAndProbabilities(...)`
- `outlier.go` + `outlier_test.go` — `OutlierScores(tree []CondensedTreeEntry, n int) []float64`
- `simplify.go` + `simplify_test.go` — `SimplifyHierarchy(...)`

**Critical design constraints:**
- CondensedTreeEntry: `Parent int, Child int, LambdaVal float64, ChildSize int`. Lambda = 1/distance.
- Condense tree: BFS from root (last dendrogram row). When both children >= minClusterSize → two new clusters. When both < minClusterSize → collapse to parent. When one < → keep larger as same cluster, collapse smaller. Track relabeling from dendrogram IDs to condensed tree cluster IDs.
- Stability: `stability(C) = sum over points in C: (lambda(point) - lambdaBirth(C))`. Root lambdaBirth = 0.
- EOM: bottom-up walk. Compare own stability vs children's sum. If children win → not selected, stability = children's sum. If self wins → selected, all descendants not selected.
- Labeling: union-find on condensed tree. Points not in any selected cluster → noise (-1). Probabilities: `prob(p) = min(lambda(p), maxLambda(C)) / maxLambda(C)`.
- GLOSH: `score(p) = (maxLambda(cluster(p)) - lambda(p)) / maxLambda(cluster(p))`. Propagate maxLambda upward. Handle +Inf lambda and zero maxLambda.
- Epsilon search: for candidates with epsilon (1/lambdaBirth) below threshold, traverse upward. Deduplicate.

**Test requirements:**
- Each component tested independently on small hand-computed examples (6-10 points)
- Condensed tree: minClusterSize=2 vs =3, all-identical points (lambda=+Inf)
- EOM: 2-cluster case, nested clusters, maxClusterSize constraint
- Labeling: noise handling, allowSingleCluster both modes, probabilities in [0,1]
- GLOSH: point at maxLambda → 0, early departure → near 1, all scores in [0,1]

**Consult reference files:**
- `reference/scikit-learn-contrib-hdbscan/hdbscan/_hdbscan_tree.pyx` — `condense_tree()`, `compute_stability()`, `get_clusters()`, `outlier_scores()`

### Agent 5: `integration` (general-purpose)

**Tasks:** 13, 14, 17, 18

Wire the full pipeline together, implement config validation, precomputed distance matrix support, edge case tests, and benchmarks. **This agent should start after the pipeline agents have made progress, or work on types/validation first.**

**Files to create:**
- `hdbscan.go` + `hdbscan_test.go` — Config, Result, `Cluster()`, `ClusterPrecomputed()`, `DefaultConfig()`
- `edge_cases_test.go` — all NFR-2.2 edge cases
- `benchmark_test.go` — benchmarks for each pipeline stage + end-to-end
- `golden_test.go` — tests that load `testdata/*.json` and compare against pipeline output

**Config struct** (with defaults):
```go
type Config struct {
    MinClusterSize              int             // default 5, must >= 2
    MinSamples                  int             // default = MinClusterSize, must >= 1
    Metric                      DistanceMetric  // default Euclidean
    ClusterSelectionMethod      string          // "eom" or "leaf", default "eom"
    Alpha                       float64         // default 1.0, must > 0
    AllowSingleCluster          bool            // default false
    ClusterSelectionEpsilon     float64         // default 0.0, must >= 0
    ClusterSelectionPersistence float64         // default 0.0, must >= 0
    MaxClusterSize              int             // default 0 (unlimited)
    ClusterSelectionEpsilonMax  float64         // default +Inf
    MatchReferenceImplementation bool           // default false
}
```

**Result struct:**
```go
type Result struct {
    Labels            []int
    Probabilities     []float64
    Stabilities       map[int]float64
    OutlierScores     []float64
    CondensedTree     []CondensedTreeEntry
    SingleLinkageTree [][4]float64
}
```

**Pipeline in Cluster():**
1. Apply defaults, validate config
2. Convert `[][]float64` input to flat `[]float64`
3. `ComputePairwiseDistances` → `ComputeCoreDistances` → `MutualReachability` → `PrimMST` → `Label` → `CondenseTree` → `ComputeStability` → select clusters (EOM or leaf) → apply epsilon search if epsilon > 0 → apply persistence simplification if threshold > 0 → `GetLabelsAndProbabilities` → `OutlierScores`
4. Return Result

**Edge cases to test (each gets its own test function):**
- n=1, n=2, all-identical points, minClusterSize > n, minSamples > n, all-noise result, single cluster with allowSingleCluster, +Inf in distance matrix

**Benchmarks:** Pairwise distances, core distances, mutual reachability, Prim's MST, condense tree, full pipeline. Sizes: 100, 500, 1000 points. Use `b.ReportAllocs()`.

## Coordination Rules

1. **Dependency order matters.** The pipeline is sequential: distance → core distance → reachability → MST → union-find/label → condensed tree → stability → selection → labeling → outliers. Agents 2-4 can start in parallel writing tests and type definitions, but implementations must respect the dependency chain in `docs/tasks.md`.

2. **Shared types go in canonical files.** If multiple agents need the same type (e.g., `CondensedTreeEntry`), the agent who creates the file that defines it owns it. Other agents import it. Coordinate to avoid conflicts:
   - `distance.go` owns `DistanceMetric`, `DistanceFunc`, metric structs
   - `condensed_tree.go` owns `CondensedTreeEntry`
   - `hdbscan.go` owns `Config`, `Result`
   - `unionfind.go` owns `UnionFind`

3. **TDD is mandatory.** Write failing tests, then implement. Every function must have tests before implementation is considered complete.

4. **Flat arrays, not slices of slices.** Matrices are `[]float64` with stride indexing. The only exception is the public `Cluster()` API which accepts `[][]float64` for ergonomics and converts internally.

5. **No premature optimization.** Phase 1 is O(n²) brute force. Don't add goroutine parallelism, spatial trees, or fancy data structures. Keep it simple and correct.

6. **Reference implementation is a guide, not gospel.** Read the Python/Cython code to understand the algorithm, but write idiomatic Go. No NumPy broadcasting patterns, no Python-isms.

7. **`gonum` is the only external dependency.** Use it for matrix operations where helpful. No CGO. No other third-party libraries.

8. **Validate every stage.** Each agent must ensure `go build ./...`, `go test ./...`, and `go vet ./...` pass for their files before declaring done. Run `golangci-lint run` if available.

## Definition of Done

Phase 1 is complete when:

- [ ] `go build ./...` succeeds with zero warnings
- [ ] `go test ./...` passes all unit tests
- [ ] `go test -race ./...` passes (no data races)
- [ ] `golangci-lint run` is clean
- [ ] Golden tests pass: output matches Python reference within tolerance (labels permutation-invariant, floats within 1e-10)
- [ ] All edge cases from NFR-2.2 pass
- [ ] Benchmarks exist and run via `go test -bench=.`
- [ ] `Cluster(data, DefaultConfig())` produces correct results on `make_blobs` and `make_moons` datasets
- [ ] No panics on any valid or edge-case input
