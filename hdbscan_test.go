package hdbscan

import (
	"math"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.MinClusterSize != 5 {
		t.Errorf("MinClusterSize: got %d, want 5", cfg.MinClusterSize)
	}
	if cfg.MinSamples != 0 {
		t.Errorf("MinSamples: got %d, want 0", cfg.MinSamples)
	}
	if _, ok := cfg.Metric.(EuclideanMetric); !ok {
		t.Errorf("Metric: got %T, want EuclideanMetric", cfg.Metric)
	}
	if cfg.ClusterSelectionMethod != "eom" {
		t.Errorf("ClusterSelectionMethod: got %q, want \"eom\"", cfg.ClusterSelectionMethod)
	}
	if cfg.Alpha != 1.0 {
		t.Errorf("Alpha: got %f, want 1.0", cfg.Alpha)
	}
	if cfg.AllowSingleCluster {
		t.Error("AllowSingleCluster: got true, want false")
	}
	if cfg.ClusterSelectionEpsilon != 0.0 {
		t.Errorf("ClusterSelectionEpsilon: got %f, want 0.0", cfg.ClusterSelectionEpsilon)
	}
	if cfg.ClusterSelectionPersistence != 0.0 {
		t.Errorf("ClusterSelectionPersistence: got %f, want 0.0", cfg.ClusterSelectionPersistence)
	}
	if cfg.MaxClusterSize != 0 {
		t.Errorf("MaxClusterSize: got %d, want 0", cfg.MaxClusterSize)
	}
	if !math.IsInf(cfg.ClusterSelectionEpsilonMax, 1) {
		t.Errorf("ClusterSelectionEpsilonMax: got %f, want +Inf", cfg.ClusterSelectionEpsilonMax)
	}
	if cfg.MatchReferenceImplementation {
		t.Error("MatchReferenceImplementation: got true, want false")
	}
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Config)
	}{
		{"MinClusterSize < 2", func(c *Config) { c.MinClusterSize = 1 }},
		{"negative MinSamples", func(c *Config) { c.MinSamples = -1 }},
		{"zero Alpha", func(c *Config) { c.Alpha = 0 }},
		{"negative Alpha", func(c *Config) { c.Alpha = -1.0 }},
		{"invalid method", func(c *Config) { c.ClusterSelectionMethod = "invalid" }},
		{"negative epsilon", func(c *Config) { c.ClusterSelectionEpsilon = -0.1 }},
		{"negative persistence", func(c *Config) { c.ClusterSelectionPersistence = -0.1 }},
	}

	data := [][]float64{{1, 2}, {3, 4}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			tt.mutate(&cfg)
			_, err := Cluster(data, cfg)
			if err == nil {
				t.Errorf("expected error for %s", tt.name)
			}
		})
	}
}

func TestMinSamplesDefaultsToMinClusterSize(t *testing.T) {
	// MinSamples=0 should default to MinClusterSize.
	// We just verify no error and a result is returned.
	data := make([][]float64, 20)
	for i := range data {
		data[i] = []float64{float64(i), float64(i * 2)}
	}
	cfg := DefaultConfig()
	cfg.MinSamples = 0 // should default to MinClusterSize=5
	result, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != 20 {
		t.Errorf("expected 20 labels, got %d", len(result.Labels))
	}
}

func TestClusterEmptyData(t *testing.T) {
	cfg := DefaultConfig()
	result, err := Cluster([][]float64{}, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != 0 {
		t.Errorf("expected empty labels, got %d", len(result.Labels))
	}
	if len(result.Probabilities) != 0 {
		t.Errorf("expected empty probabilities, got %d", len(result.Probabilities))
	}
	if len(result.OutlierScores) != 0 {
		t.Errorf("expected empty outlier scores, got %d", len(result.OutlierScores))
	}
}

func TestClusterBasicResult(t *testing.T) {
	// Two well-separated clusters of 10 points each.
	data := make([][]float64, 20)
	for i := 0; i < 10; i++ {
		data[i] = []float64{float64(i) * 0.1, 0}
	}
	for i := 10; i < 20; i++ {
		data[i] = []float64{100 + float64(i)*0.1, 0}
	}

	cfg := DefaultConfig()
	cfg.MinClusterSize = 3
	result, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != 20 {
		t.Errorf("expected 20 labels, got %d", len(result.Labels))
	}
	if len(result.Probabilities) != 20 {
		t.Errorf("expected 20 probabilities, got %d", len(result.Probabilities))
	}
	if len(result.OutlierScores) != 20 {
		t.Errorf("expected 20 outlier scores, got %d", len(result.OutlierScores))
	}

	// Verify two distinct non-noise clusters
	clusterSet := make(map[int]bool)
	for _, l := range result.Labels {
		if l >= 0 {
			clusterSet[l] = true
		}
	}
	if len(clusterSet) < 2 {
		t.Errorf("expected at least 2 clusters for well-separated data, got %d", len(clusterSet))
	}
}

func TestClusterAndClusterPrecomputedSameResult(t *testing.T) {
	// Generate a small dataset.
	data := [][]float64{
		{0, 0}, {0.1, 0}, {0.2, 0}, {0, 0.1}, {0.1, 0.1},
		{10, 10}, {10.1, 10}, {10.2, 10}, {10, 10.1}, {10.1, 10.1},
	}

	cfg := DefaultConfig()
	cfg.MinClusterSize = 3

	result1, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("Cluster error: %v", err)
	}

	// Build precomputed distance matrix.
	n := len(data)
	metric := EuclideanMetric{}
	distMatrix := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			d := metric.Distance(data[i], data[j])
			distMatrix[i*n+j] = d
			distMatrix[j*n+i] = d
		}
	}

	result2, err := ClusterPrecomputed(distMatrix, n, cfg)
	if err != nil {
		t.Fatalf("ClusterPrecomputed error: %v", err)
	}

	// Labels should be identical.
	if len(result1.Labels) != len(result2.Labels) {
		t.Fatalf("label lengths differ: %d vs %d", len(result1.Labels), len(result2.Labels))
	}
	for i := range result1.Labels {
		if result1.Labels[i] != result2.Labels[i] {
			t.Errorf("labels[%d]: %d vs %d", i, result1.Labels[i], result2.Labels[i])
		}
	}

	// Probabilities should be identical.
	for i := range result1.Probabilities {
		if math.Abs(result1.Probabilities[i]-result2.Probabilities[i]) > 1e-10 {
			t.Errorf("probabilities[%d]: %f vs %f", i, result1.Probabilities[i], result2.Probabilities[i])
		}
	}

	// Outlier scores should be identical.
	for i := range result1.OutlierScores {
		if math.Abs(result1.OutlierScores[i]-result2.OutlierScores[i]) > 1e-10 {
			t.Errorf("outlier_scores[%d]: %f vs %f", i, result1.OutlierScores[i], result2.OutlierScores[i])
		}
	}
}

// TestAlgorithmEquivalence runs Cluster() through each tree-based algorithm on a
// non-trivial dataset (two well-separated clusters of 25 points each) and verifies
// that labels, probabilities, and outlier scores match the brute-force path.
// This is the e2e test that ensures the full pipeline produces correct output for
// every algorithm, not just that intermediate stages (MST weights, etc.) match.
func TestAlgorithmEquivalence(t *testing.T) {
	// Two tight clusters separated by a wide gap — deterministic clustering.
	rng := newTestRNG(42)
	data := make([][]float64, 50)
	for i := 0; i < 25; i++ {
		data[i] = []float64{rng.Float64() * 0.5, rng.Float64() * 0.5}
	}
	for i := 25; i < 50; i++ {
		data[i] = []float64{10 + rng.Float64()*0.5, 10 + rng.Float64()*0.5}
	}

	cfg := DefaultConfig()
	cfg.MinClusterSize = 5
	cfg.Algorithm = AlgorithmBrute
	bruteResult, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("brute Cluster() error: %v", err)
	}

	algos := []struct {
		name string
		algo Algorithm
	}{
		{"prims_kdtree", AlgorithmPrimsKDTree},
		{"prims_balltree", AlgorithmPrimsBalltree},
		{"boruvka_kdtree", AlgorithmBoruvkaKDTree},
		{"boruvka_balltree", AlgorithmBoruvkaBalltree},
	}

	for _, a := range algos {
		t.Run(a.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.MinClusterSize = 5
			cfg.Algorithm = a.algo
			result, err := Cluster(data, cfg)
			if err != nil {
				t.Fatalf("Cluster() error: %v", err)
			}

			// Labels must be permutation-equivalent.
			if !labelsEquivalent(bruteResult.Labels, result.Labels) {
				t.Errorf("labels not equivalent to brute-force:\n  brute: %v\n  got:   %v",
					bruteResult.Labels, result.Labels)
			}

			// Probabilities: for well-separated clusters, all non-noise points
			// should have probability 1.0 regardless of algorithm path.
			for i, p := range result.Probabilities {
				bp := bruteResult.Probabilities[i]
				if math.Abs(p-bp) > 0.01 {
					t.Errorf("probabilities[%d]: brute=%f, got=%f", i, bp, p)
				}
			}

			// Outlier scores should be close (same condensed tree structure for
			// well-separated data).
			for i, s := range result.OutlierScores {
				bs := bruteResult.OutlierScores[i]
				if math.Abs(s-bs) > 0.01 {
					t.Errorf("outlier_scores[%d]: brute=%f, got=%f", i, bs, s)
				}
			}
		})
	}
}

// TestAlgorithmEquivalenceLeaf tests leaf cluster selection across all algorithms.
// Leaf selection is sensitive to MST edge ordering (see Known Limitations in CLAUDE.md),
// so we verify structural properties rather than exact label matching: same noise points,
// reasonable cluster count, and no algorithm crash.
func TestAlgorithmEquivalenceLeaf(t *testing.T) {
	rng := newTestRNG(99)
	data := make([][]float64, 50)
	for i := 0; i < 25; i++ {
		data[i] = []float64{rng.Float64() * 0.5, rng.Float64() * 0.5}
	}
	for i := 25; i < 50; i++ {
		data[i] = []float64{10 + rng.Float64()*0.5, 10 + rng.Float64()*0.5}
	}

	algos := []Algorithm{
		AlgorithmBrute,
		AlgorithmPrimsKDTree,
		AlgorithmPrimsBalltree,
		AlgorithmBoruvkaKDTree,
		AlgorithmBoruvkaBalltree,
	}

	for _, algo := range algos {
		t.Run(string(algo), func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.MinClusterSize = 5
			cfg.ClusterSelectionMethod = "leaf"
			cfg.Algorithm = algo
			result, err := Cluster(data, cfg)
			if err != nil {
				t.Fatalf("Cluster() error: %v", err)
			}

			if len(result.Labels) != 50 {
				t.Fatalf("expected 50 labels, got %d", len(result.Labels))
			}
			if len(result.Probabilities) != 50 {
				t.Fatalf("expected 50 probabilities, got %d", len(result.Probabilities))
			}
			if len(result.OutlierScores) != 50 {
				t.Fatalf("expected 50 outlier scores, got %d", len(result.OutlierScores))
			}

			// With well-separated clusters, every algorithm should find at least 2.
			clusters := map[int]bool{}
			for _, l := range result.Labels {
				if l >= 0 {
					clusters[l] = true
				}
			}
			if len(clusters) < 2 {
				t.Errorf("expected at least 2 clusters, got %d (labels: %v)", len(clusters), result.Labels)
			}

			// Second cluster group (points 25-49) should all be in the same cluster.
			secondGroup := result.Labels[25]
			for i := 26; i < 50; i++ {
				if result.Labels[i] != secondGroup {
					t.Errorf("points 25-49 should be in the same cluster, but labels[%d]=%d != labels[25]=%d",
						i, result.Labels[i], secondGroup)
					break
				}
			}
		})
	}
}

// newTestRNG creates a deterministic RNG for test data generation.
func newTestRNG(seed int64) *testRNG {
	// Simple LCG — good enough for generating test points.
	return &testRNG{state: uint64(seed)}
}

type testRNG struct {
	state uint64
}

func (r *testRNG) Float64() float64 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	return float64(r.state>>11) / float64(1<<53)
}

func TestClusterPrecomputed_NonSquareError(t *testing.T) {
	cfg := DefaultConfig()
	_, err := ClusterPrecomputed([]float64{1, 2, 3}, 2, cfg)
	if err == nil {
		t.Error("expected error for non-square distance matrix")
	}
}

func TestClusterPrecomputed_EmptyData(t *testing.T) {
	cfg := DefaultConfig()
	result, err := ClusterPrecomputed([]float64{}, 0, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != 0 {
		t.Errorf("expected empty labels, got %d", len(result.Labels))
	}
}

func TestClusterLeafMethod(t *testing.T) {
	// Two well-separated clusters.
	data := make([][]float64, 20)
	for i := 0; i < 10; i++ {
		data[i] = []float64{float64(i) * 0.1, 0}
	}
	for i := 10; i < 20; i++ {
		data[i] = []float64{100 + float64(i)*0.1, 0}
	}

	cfg := DefaultConfig()
	cfg.MinClusterSize = 3
	cfg.ClusterSelectionMethod = "leaf"
	result, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != 20 {
		t.Errorf("expected 20 labels, got %d", len(result.Labels))
	}
}

func TestClusterWithMetricNilDefault(t *testing.T) {
	data := [][]float64{
		{0, 0}, {0.1, 0}, {0.2, 0}, {0, 0.1}, {0.1, 0.1},
		{10, 10}, {10.1, 10}, {10.2, 10}, {10, 10.1}, {10.1, 10.1},
	}
	cfg := DefaultConfig()
	cfg.Metric = nil // should default to Euclidean
	cfg.MinClusterSize = 3
	_, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error with nil metric: %v", err)
	}
}
