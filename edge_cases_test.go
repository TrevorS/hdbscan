package hdbscan

import (
	"math"
	"testing"
)

func TestEdgeCase_SinglePoint(t *testing.T) {
	data := [][]float64{{1.0, 2.0}}
	cfg := DefaultConfig()
	result, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != 1 {
		t.Fatalf("expected 1 label, got %d", len(result.Labels))
	}
	// Single point should be noise.
	if result.Labels[0] != -1 {
		t.Errorf("expected label -1 for single point, got %d", result.Labels[0])
	}
	if len(result.Probabilities) != 1 {
		t.Errorf("expected 1 probability, got %d", len(result.Probabilities))
	}
	if len(result.OutlierScores) != 1 {
		t.Errorf("expected 1 outlier score, got %d", len(result.OutlierScores))
	}
}

func TestEdgeCase_TwoPoints(t *testing.T) {
	data := [][]float64{{0, 0}, {1, 0}}
	cfg := DefaultConfig()
	cfg.MinClusterSize = 2
	result, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != 2 {
		t.Fatalf("expected 2 labels, got %d", len(result.Labels))
	}
	// Both should either be in one cluster or noise -- no panic is the key test.
}

func TestEdgeCase_AllIdenticalPoints(t *testing.T) {
	data := make([][]float64, 10)
	for i := range data {
		data[i] = []float64{5.0, 5.0}
	}
	cfg := DefaultConfig()
	cfg.MinClusterSize = 3
	result, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != 10 {
		t.Fatalf("expected 10 labels, got %d", len(result.Labels))
	}
	// Verify no NaN in probabilities or outlier scores.
	for i, p := range result.Probabilities {
		if math.IsNaN(p) {
			t.Errorf("NaN probability at index %d", i)
		}
	}
	for i, s := range result.OutlierScores {
		if math.IsNaN(s) {
			t.Errorf("NaN outlier score at index %d", i)
		}
	}
}

func TestEdgeCase_MinClusterSizeGreaterThanN(t *testing.T) {
	data := [][]float64{{0, 0}, {1, 0}, {2, 0}}
	cfg := DefaultConfig()
	cfg.MinClusterSize = 10 // bigger than n=3
	result, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// All points should be noise since no cluster can form.
	for i, l := range result.Labels {
		if l != -1 {
			t.Errorf("expected noise (-1) at index %d, got %d", i, l)
		}
	}
}

func TestEdgeCase_MinSamplesGreaterThanN(t *testing.T) {
	data := [][]float64{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}}
	cfg := DefaultConfig()
	cfg.MinClusterSize = 2
	cfg.MinSamples = 100 // much bigger than n=5, will be clamped to 4
	result, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != 5 {
		t.Fatalf("expected 5 labels, got %d", len(result.Labels))
	}
}

func TestEdgeCase_AllNoise(t *testing.T) {
	// Points too sparse for any cluster to form with high minClusterSize.
	data := [][]float64{
		{0, 0}, {100, 100}, {200, 200}, {300, 300}, {400, 400},
		{500, 500}, {600, 600}, {700, 700}, {800, 800}, {900, 900},
	}
	cfg := DefaultConfig()
	cfg.MinClusterSize = 5
	result, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, l := range result.Labels {
		if l != -1 {
			t.Errorf("expected all noise, but index %d has label %d", i, l)
		}
	}
	for i, p := range result.Probabilities {
		if p != 0 {
			t.Errorf("expected 0 probability for noise at index %d, got %f", i, p)
		}
	}
}

func TestEdgeCase_SingleClusterWithAllowSingleCluster(t *testing.T) {
	// A tight cluster of points; with allowSingleCluster, should form one cluster.
	data := make([][]float64, 20)
	for i := range data {
		data[i] = []float64{float64(i) * 0.01, float64(i) * 0.01}
	}
	cfg := DefaultConfig()
	cfg.MinClusterSize = 5
	cfg.AllowSingleCluster = true
	result, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != 20 {
		t.Fatalf("expected 20 labels, got %d", len(result.Labels))
	}
	// No panics or NaN.
	for i, p := range result.Probabilities {
		if math.IsNaN(p) {
			t.Errorf("NaN probability at index %d", i)
		}
	}
	for i, s := range result.OutlierScores {
		if math.IsNaN(s) {
			t.Errorf("NaN outlier score at index %d", i)
		}
	}
}

func TestEdgeCase_InfInDistanceMatrix(t *testing.T) {
	n := 5
	// Build a distance matrix with some +Inf entries.
	distMatrix := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				distMatrix[i*n+j] = 0
			} else if (i == 0 && j == 4) || (i == 4 && j == 0) {
				distMatrix[i*n+j] = math.Inf(1) // missing distance
			} else {
				distMatrix[i*n+j] = math.Abs(float64(i) - float64(j))
			}
		}
	}

	cfg := DefaultConfig()
	cfg.MinClusterSize = 2
	result, err := ClusterPrecomputed(distMatrix, n, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != n {
		t.Fatalf("expected %d labels, got %d", n, len(result.Labels))
	}
	// No panics or NaN.
	for i, p := range result.Probabilities {
		if math.IsNaN(p) {
			t.Errorf("NaN probability at index %d", i)
		}
	}
}

func TestEdgeCase_ThreePointsMinClusterSize2(t *testing.T) {
	data := [][]float64{{0, 0}, {0.1, 0}, {100, 100}}
	cfg := DefaultConfig()
	cfg.MinClusterSize = 2
	result, err := Cluster(data, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Labels) != 3 {
		t.Fatalf("expected 3 labels, got %d", len(result.Labels))
	}
}

// threeClusterData returns 30 points in 3 well-separated groups of 10.
func threeClusterData() [][]float64 {
	data := make([][]float64, 30)
	for i := 0; i < 10; i++ {
		data[i] = []float64{float64(i) * 0.1, 0}
	}
	for i := 10; i < 20; i++ {
		data[i] = []float64{50 + float64(i)*0.1, 0}
	}
	for i := 20; i < 30; i++ {
		data[i] = []float64{100 + float64(i)*0.1, 0}
	}
	return data
}

func TestEdgeCase_ProbabilitiesInRange(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinClusterSize = 3
	result, err := Cluster(threeClusterData(), cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, p := range result.Probabilities {
		if p < 0 || p > 1.0+1e-10 {
			t.Errorf("probability[%d] = %f out of range [0, 1]", i, p)
		}
	}
}

func TestEdgeCase_OutlierScoresInRange(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinClusterSize = 3
	result, err := Cluster(threeClusterData(), cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, s := range result.OutlierScores {
		if s < 0 || s > 1.0+1e-10 {
			t.Errorf("outlier_score[%d] = %f out of range [0, 1]", i, s)
		}
	}
}
