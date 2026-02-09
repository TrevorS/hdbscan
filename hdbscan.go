package hdbscan

import (
	"errors"
	"fmt"
	"math"
	"runtime"
)

// Algorithm selects the MST construction strategy.
type Algorithm string

const (
	AlgorithmAuto            Algorithm = "auto"
	AlgorithmBrute           Algorithm = "brute"
	AlgorithmPrimsKDTree     Algorithm = "prims_kdtree"
	AlgorithmPrimsBalltree   Algorithm = "prims_balltree"
	AlgorithmBoruvkaKDTree   Algorithm = "boruvka_kdtree"
	AlgorithmBoruvkaBalltree Algorithm = "boruvka_balltree"
)

// Config controls HDBSCAN clustering behavior.
// Start with [DefaultConfig] and override the fields you need.
type Config struct {
	// MinClusterSize is the smallest group of points considered a cluster.
	// Smaller values find more clusters; larger values find fewer, denser ones.
	// Must be >= 2. Default: 5.
	MinClusterSize int

	// MinSamples controls noise sensitivity. Higher values label more points as
	// noise but may miss sparser clusters. Set to 0 to default to MinClusterSize.
	// Must be >= 0. Default: MinClusterSize.
	MinSamples int

	// Metric is the distance function used to measure point similarity.
	// Built-in: EuclideanMetric, ManhattanMetric, CosineMetric, ChebyshevMetric,
	// MinkowskiMetric. Use DistanceFunc to wrap a custom function.
	// Default: EuclideanMetric.
	Metric DistanceMetric

	// ClusterSelectionMethod chooses how flat clusters are extracted from the
	// condensed tree. "eom" (Excess of Mass) maximizes cluster stability.
	// "leaf" selects the leaves, producing many small homogeneous clusters.
	// Default: "eom".
	ClusterSelectionMethod string

	// Alpha scales pairwise distances before computing mutual reachability.
	// Values < 1.0 produce more aggressive clustering (fewer noise points).
	// Must be > 0. Default: 1.0.
	Alpha float64

	// AllowSingleCluster permits the algorithm to return all points in one
	// cluster rather than splitting into subclusters. Default: false.
	AllowSingleCluster bool

	// ClusterSelectionEpsilon sets a minimum distance threshold below which
	// clusters will not be split further. Prevents over-segmentation in dense
	// regions. 0 means no threshold. Must be >= 0. Default: 0.0.
	ClusterSelectionEpsilon float64

	// ClusterSelectionPersistence removes clusters whose persistence (lifespan
	// in the condensed tree) is below this threshold, simplifying the hierarchy.
	// 0 means no simplification. Must be >= 0. Default: 0.0.
	ClusterSelectionPersistence float64

	// MaxClusterSize forces subclusters to be selected over a parent cluster
	// when the parent exceeds this size. 0 means unlimited. Default: 0.
	MaxClusterSize int

	// ClusterSelectionEpsilonMax is an upper bound on epsilon for EOM selection.
	// Clusters with epsilon above this threshold are split into subclusters.
	// Default: +Inf (no upper bound).
	ClusterSelectionEpsilonMax float64

	// MatchReferenceImplementation enables edge-case behaviors that exactly
	// match the Python scikit-learn-contrib/hdbscan library, at the cost of
	// some API cleanliness. Default: false.
	MatchReferenceImplementation bool

	// Algorithm selects the MST construction strategy.
	// "auto" picks the best algorithm based on metric and dimensionality.
	// "brute" uses the full distance matrix (O(n²) memory).
	// "prims_kdtree"/"prims_balltree" use Prim's without a full matrix.
	// "boruvka_kdtree"/"boruvka_balltree" use dual-tree Borůvka (fastest for
	// low-dimensional data). Default: "auto".
	Algorithm Algorithm

	// LeafSize controls the maximum number of points in a spatial tree leaf node.
	// Larger values trade query precision for faster tree construction.
	// Only used with tree-based algorithms. Default: 40.
	LeafSize int

	// Workers controls the number of goroutines for parallelizable stages
	// (pairwise distances, core distances, mutual reachability). Only affects
	// the brute-force algorithm path. 0 means use runtime.NumCPU().
	// Default: 0 (auto).
	Workers int
}

// Result contains the output of HDBSCAN clustering.
type Result struct {
	// Labels assigns each point to a cluster (0-indexed cluster ID) or -1 for
	// noise (points not assigned to any cluster).
	Labels []int

	// Probabilities indicates how strongly each point belongs to its assigned
	// cluster, in [0, 1]. Noise points have probability 0.
	Probabilities []float64

	// Stabilities maps cluster IDs to their stability values. Higher stability
	// means the cluster persists across a wider range of density thresholds.
	Stabilities map[int]float64

	// OutlierScores is the GLOSH (Global-Local Outlier Score from Hierarchies)
	// score for each point, in [0, 1]. Values near 0 indicate inliers; values
	// near 1 indicate strong outliers.
	OutlierScores []float64

	// CondensedTree is the internal condensed cluster hierarchy. Useful for
	// visualization or custom post-processing.
	CondensedTree []CondensedTreeEntry

	// SingleLinkageTree is the full single-linkage dendrogram in scipy format:
	// each row is [left, right, distance, size]. Internal cluster IDs start at n.
	SingleLinkageTree [][4]float64
}

// DefaultConfig returns a Config with reasonable defaults.
func DefaultConfig() Config {
	return Config{
		MinClusterSize:             5,
		Metric:                     EuclideanMetric{},
		ClusterSelectionMethod:     "eom",
		Alpha:                      1.0,
		ClusterSelectionEpsilonMax: math.Inf(1),
	}
}

// validateConfig checks that cfg fields are valid and returns a descriptive error if not.
func validateConfig(cfg *Config) error {
	if cfg.MinClusterSize < 2 {
		return fmt.Errorf("hdbscan: MinClusterSize must be >= 2, got %d", cfg.MinClusterSize)
	}
	if cfg.MinSamples < 0 {
		return fmt.Errorf("hdbscan: MinSamples must be >= 0 (0 means default to MinClusterSize), got %d", cfg.MinSamples)
	}
	if cfg.Alpha <= 0 {
		return fmt.Errorf("hdbscan: Alpha must be > 0, got %f", cfg.Alpha)
	}
	if cfg.ClusterSelectionMethod != "eom" && cfg.ClusterSelectionMethod != "leaf" {
		return fmt.Errorf("hdbscan: ClusterSelectionMethod must be \"eom\" or \"leaf\", got %q", cfg.ClusterSelectionMethod)
	}
	if cfg.ClusterSelectionEpsilon < 0 {
		return fmt.Errorf("hdbscan: ClusterSelectionEpsilon must be >= 0, got %f", cfg.ClusterSelectionEpsilon)
	}
	if cfg.ClusterSelectionPersistence < 0 {
		return fmt.Errorf("hdbscan: ClusterSelectionPersistence must be >= 0, got %f", cfg.ClusterSelectionPersistence)
	}
	switch cfg.Algorithm {
	case AlgorithmAuto, AlgorithmBrute,
		AlgorithmPrimsKDTree, AlgorithmPrimsBalltree,
		AlgorithmBoruvkaKDTree, AlgorithmBoruvkaBalltree:
		// valid
	default:
		return fmt.Errorf("hdbscan: invalid Algorithm %q", cfg.Algorithm)
	}
	if cfg.LeafSize < 1 {
		return fmt.Errorf("hdbscan: LeafSize must be >= 1, got %d", cfg.LeafSize)
	}
	return nil
}

// applyDefaults fills in zero-valued config fields with their defaults.
func applyDefaults(cfg *Config) {
	if cfg.MinSamples == 0 {
		cfg.MinSamples = cfg.MinClusterSize
	}
	if cfg.Metric == nil {
		cfg.Metric = EuclideanMetric{}
	}
	if cfg.ClusterSelectionEpsilonMax == 0 {
		cfg.ClusterSelectionEpsilonMax = math.Inf(1)
	}
	if cfg.Algorithm == "" {
		cfg.Algorithm = AlgorithmAuto
	}
	if cfg.LeafSize == 0 {
		cfg.LeafSize = 40
	}
	if cfg.Workers == 0 {
		cfg.Workers = runtime.NumCPU()
	}
}

// emptyResult returns a Result with empty/zero-length slices for n data points.
// When n is 0, all slices are non-nil but empty.
func emptyResult(n int) *Result {
	return &Result{
		Labels:        make([]int, n),
		Probabilities: make([]float64, n),
		Stabilities:   map[int]float64{},
		OutlierScores: make([]float64, n),
	}
}

// Cluster performs HDBSCAN clustering on the given data.
// Each element is a point (float64 slice); all points must have the same
// dimensionality. Returns an error if the config is invalid.
func Cluster(data [][]float64, cfg Config) (*Result, error) {
	applyDefaults(&cfg)
	if err := validateConfig(&cfg); err != nil {
		return nil, err
	}

	n := len(data)
	if n == 0 {
		return emptyResult(0), nil
	}

	dims := len(data[0])
	flatData := make([]float64, n*dims)
	for i, row := range data {
		copy(flatData[i*dims:], row)
	}

	algo, err := selectAlgorithm(cfg, n, dims)
	if err != nil {
		return nil, err
	}

	switch algo {
	case AlgorithmPrimsKDTree, AlgorithmPrimsBalltree:
		return clusterPrimsTree(flatData, n, dims, cfg, algo)
	case AlgorithmBoruvkaKDTree, AlgorithmBoruvkaBalltree:
		return clusterBoruvka(flatData, n, dims, cfg, algo)
	default:
		// AlgorithmBrute: use full distance matrix.
		distMatrix := ComputePairwiseDistancesParallel(flatData, n, dims, cfg.Metric, cfg.Workers)
		return clusterFromDistMatrix(distMatrix, n, cfg)
	}
}

// ClusterPrecomputed performs HDBSCAN on a precomputed distance matrix.
// distMatrix is a flat []float64 of length n*n in row-major order, where
// distMatrix[i*n+j] is the distance between points i and j. The Config.Metric
// field is ignored since distances are already computed.
func ClusterPrecomputed(distMatrix []float64, n int, cfg Config) (*Result, error) {
	applyDefaults(&cfg)
	if err := validateConfig(&cfg); err != nil {
		return nil, err
	}

	if len(distMatrix) != n*n {
		return nil, fmt.Errorf("hdbscan: distMatrix length %d does not match n*n = %d (n=%d)", len(distMatrix), n*n, n)
	}

	if n == 0 {
		return emptyResult(0), nil
	}

	return clusterFromDistMatrix(distMatrix, n, cfg)
}

// clusterPrimsTree runs the HDBSCAN pipeline using tree-accelerated core distances
// and matrix-free Prim's MST (O(n²) time, O(n) memory).
func clusterPrimsTree(flatData []float64, n, dims int, cfg Config, algo Algorithm) (*Result, error) {
	minSamples := cfg.MinSamples
	if minSamples > n-1 {
		minSamples = n - 1
	}
	if minSamples < 1 && n > 1 {
		return nil, errors.New("hdbscan: MinSamples must be >= 1 after defaulting")
	}
	if n == 1 {
		r := emptyResult(1)
		r.Labels[0] = -1
		return r, nil
	}

	var tree SpatialTree
	switch algo {
	case AlgorithmPrimsKDTree:
		tree = NewKDTree(flatData, n, dims, cfg.Metric, cfg.LeafSize)
	default:
		tree = NewBallTree(flatData, n, dims, cfg.Metric, cfg.LeafSize)
	}

	coreDistances := ComputeCoreDistancesTree(tree, minSamples)
	mstEdges := PrimMSTVector(flatData, n, dims, coreDistances, cfg.Metric, cfg.Alpha)
	return clusterFromMST(mstEdges, n, cfg)
}

// clusterBoruvka runs the HDBSCAN pipeline using dual-tree Borůvka MST construction.
func clusterBoruvka(flatData []float64, n, dims int, cfg Config, algo Algorithm) (*Result, error) {
	if n == 1 {
		r := emptyResult(1)
		r.Labels[0] = -1
		return r, nil
	}

	minSamples := cfg.MinSamples
	if minSamples > n-1 {
		minSamples = n - 1
	}

	var tree BoruvkaTree
	switch algo {
	case AlgorithmBoruvkaKDTree:
		tree = NewKDTree(flatData, n, dims, cfg.Metric, cfg.LeafSize)
	default:
		tree = NewBallTree(flatData, n, dims, cfg.Metric, cfg.LeafSize)
	}

	var mstEdges [][3]float64
	switch algo {
	case AlgorithmBoruvkaKDTree:
		b := NewKDTreeBoruvka(tree, cfg.Metric, minSamples, cfg.Alpha)
		mstEdges, _ = b.SpanningTree()
	default:
		b := NewBallTreeBoruvka(tree, cfg.Metric, minSamples, cfg.Alpha)
		mstEdges, _ = b.SpanningTree()
	}

	return clusterFromMST(mstEdges, n, cfg)
}

// clusterFromMST runs the HDBSCAN pipeline from MST edges onward
// (Label → CondenseTree → selection → labels/probabilities → outlier scores).
func clusterFromMST(mstEdges [][3]float64, n int, cfg Config) (*Result, error) {
	dendrogram := Label(mstEdges, n)
	condensedTree := CondenseTree(dendrogram, cfg.MinClusterSize)

	if condensedTree == nil {
		r := emptyResult(n)
		for i := range r.Labels {
			r.Labels[i] = -1
		}
		r.SingleLinkageTree = dendrogram
		return r, nil
	}

	stability := ComputeStability(condensedTree)

	if cfg.ClusterSelectionPersistence > 0 {
		condensedTree = SimplifyHierarchy(condensedTree, cfg.ClusterSelectionPersistence)
		stability = ComputeStability(condensedTree)
	}

	var selectedClusters map[int]bool
	var updatedStability map[int]float64
	switch cfg.ClusterSelectionMethod {
	case "eom":
		selectedClusters, updatedStability = SelectClustersEOM(
			condensedTree, stability,
			cfg.AllowSingleCluster, cfg.MaxClusterSize, cfg.ClusterSelectionEpsilonMax,
		)
		if cfg.ClusterSelectionEpsilon > 0 {
			selectedClusters = EpsilonSearch(condensedTree, selectedClusters,
				cfg.ClusterSelectionEpsilon, cfg.AllowSingleCluster)
		}
	case "leaf":
		selectedClusters = SelectClustersLeaf(condensedTree, cfg.ClusterSelectionEpsilon)
		updatedStability = stability
	}

	labels, probabilities := GetLabelsAndProbabilities(
		condensedTree, selectedClusters, n,
		cfg.AllowSingleCluster, cfg.ClusterSelectionEpsilon,
		cfg.MatchReferenceImplementation,
	)

	outlierScores := OutlierScores(condensedTree, n)

	return &Result{
		Labels:            labels,
		Probabilities:     probabilities,
		Stabilities:       updatedStability,
		OutlierScores:     outlierScores,
		CondensedTree:     condensedTree,
		SingleLinkageTree: dendrogram,
	}, nil
}

// clusterFromDistMatrix runs the HDBSCAN pipeline from a precomputed distance matrix.
func clusterFromDistMatrix(distMatrix []float64, n int, cfg Config) (*Result, error) {
	minSamples := cfg.MinSamples
	if minSamples > n-1 {
		minSamples = n - 1
	}
	if minSamples < 1 && n > 1 {
		return nil, errors.New("hdbscan: MinSamples must be >= 1 after defaulting")
	}

	if n == 1 {
		r := emptyResult(1)
		r.Labels[0] = -1
		return r, nil
	}

	coreDistances := ComputeCoreDistancesParallel(distMatrix, n, minSamples, cfg.Workers)
	mrMatrix := MutualReachabilityParallel(distMatrix, coreDistances, n, cfg.Alpha, cfg.Workers)
	mstEdges := PrimMST(mrMatrix, n)
	return clusterFromMST(mstEdges, n, cfg)
}
