package hdbscan

import (
	"testing"
)

func TestComputeCoreDistancesTree_MatchesBruteForce_KDTree(t *testing.T) {
	data := []float64{
		0, 0,
		3, 0,
		0, 4,
		3, 4,
		1.5, 2,
	}
	n, dims := 5, 2
	metric := EuclideanMetric{}

	distMatrix := ComputePairwiseDistances(data, n, dims, metric)

	for minSamples := 1; minSamples <= n-1; minSamples++ {
		expected := ComputeCoreDistances(distMatrix, n, minSamples)
		tree := NewKDTree(data, n, dims, metric, 2)
		got := ComputeCoreDistancesTree(tree, minSamples)

		for i := 0; i < n; i++ {
			if !almostEqual(got[i], expected[i], floatTol) {
				t.Errorf("KDTree minSamples=%d: core[%d] = %v, want %v", minSamples, i, got[i], expected[i])
			}
		}
	}
}

func TestComputeCoreDistancesTree_MatchesBruteForce_BallTree(t *testing.T) {
	data := []float64{
		0, 0,
		3, 0,
		0, 4,
		3, 4,
		1.5, 2,
	}
	n, dims := 5, 2
	metric := EuclideanMetric{}

	distMatrix := ComputePairwiseDistances(data, n, dims, metric)

	for minSamples := 1; minSamples <= n-1; minSamples++ {
		expected := ComputeCoreDistances(distMatrix, n, minSamples)
		tree := NewBallTree(data, n, dims, metric, 2)
		got := ComputeCoreDistancesTree(tree, minSamples)

		for i := 0; i < n; i++ {
			if !almostEqual(got[i], expected[i], floatTol) {
				t.Errorf("BallTree minSamples=%d: core[%d] = %v, want %v", minSamples, i, got[i], expected[i])
			}
		}
	}
}

func TestComputeCoreDistancesTree_Collinear(t *testing.T) {
	// 5 points on x-axis: (0,0), (1,0), (2,0), (3,0), (4,0)
	data := []float64{0, 0, 1, 0, 2, 0, 3, 0, 4, 0}
	n, dims := 5, 2
	metric := EuclideanMetric{}

	distMatrix := ComputePairwiseDistances(data, n, dims, metric)

	for minSamples := 1; minSamples <= 4; minSamples++ {
		expected := ComputeCoreDistances(distMatrix, n, minSamples)

		for _, builder := range []struct {
			name string
			tree SpatialTree
		}{
			{"KDTree", NewKDTree(data, n, dims, metric, 2)},
			{"BallTree", NewBallTree(data, n, dims, metric, 2)},
		} {
			got := ComputeCoreDistancesTree(builder.tree, minSamples)
			for i := 0; i < n; i++ {
				if !almostEqual(got[i], expected[i], floatTol) {
					t.Errorf("%s minSamples=%d: core[%d] = %v, want %v",
						builder.name, minSamples, i, got[i], expected[i])
				}
			}
		}
	}
}

func TestComputeCoreDistancesTree_SinglePoint(t *testing.T) {
	data := []float64{5, 5}
	tree := NewKDTree(data, 1, 2, EuclideanMetric{}, 10)

	core := ComputeCoreDistancesTree(tree, 5)
	if len(core) != 1 {
		t.Fatalf("expected length 1, got %d", len(core))
	}
	if core[0] != 0 {
		t.Errorf("expected 0 for single point, got %v", core[0])
	}
}

func TestComputeCoreDistancesTree_TwoPoints(t *testing.T) {
	data := []float64{0, 0, 3, 4}
	n, dims := 2, 2
	metric := EuclideanMetric{}

	distMatrix := ComputePairwiseDistances(data, n, dims, metric)
	expected := ComputeCoreDistances(distMatrix, n, 1)

	tree := NewKDTree(data, n, dims, metric, 10)
	got := ComputeCoreDistancesTree(tree, 1)

	for i := 0; i < n; i++ {
		if !almostEqual(got[i], expected[i], floatTol) {
			t.Errorf("core[%d] = %v, want %v", i, got[i], expected[i])
		}
	}
}

func TestComputeCoreDistancesTree_MinSamplesClampedToNMinus1(t *testing.T) {
	data := []float64{0, 0, 1, 1, 2, 2}
	n, dims := 3, 2
	metric := EuclideanMetric{}

	distMatrix := ComputePairwiseDistances(data, n, dims, metric)
	expected := ComputeCoreDistances(distMatrix, n, 10) // clamped to n-1=2

	tree := NewKDTree(data, n, dims, metric, 10)
	got := ComputeCoreDistancesTree(tree, 10)

	for i := 0; i < n; i++ {
		if !almostEqual(got[i], expected[i], floatTol) {
			t.Errorf("core[%d] = %v, want %v", i, got[i], expected[i])
		}
	}
}

func TestComputeCoreDistancesTree_IdenticalPoints(t *testing.T) {
	data := []float64{1, 1, 1, 1, 1, 1}
	n, dims := 3, 2
	metric := EuclideanMetric{}

	for _, builder := range []struct {
		name string
		tree SpatialTree
	}{
		{"KDTree", NewKDTree(data, n, dims, metric, 2)},
		{"BallTree", NewBallTree(data, n, dims, metric, 2)},
	} {
		core := ComputeCoreDistancesTree(builder.tree, 2)
		for i := 0; i < n; i++ {
			if core[i] != 0 {
				t.Errorf("%s: core[%d] = %v, want 0 for identical points", builder.name, i, core[i])
			}
		}
	}
}

func TestComputeCoreDistancesTree_Manhattan(t *testing.T) {
	data := []float64{0, 0, 3, 0, 0, 4, 3, 4, 1.5, 2}
	n, dims := 5, 2
	metric := ManhattanMetric{}

	distMatrix := ComputePairwiseDistances(data, n, dims, metric)

	for minSamples := 1; minSamples <= 3; minSamples++ {
		expected := ComputeCoreDistances(distMatrix, n, minSamples)
		tree := NewKDTree(data, n, dims, metric, 2)
		got := ComputeCoreDistancesTree(tree, minSamples)

		for i := 0; i < n; i++ {
			if !almostEqual(got[i], expected[i], floatTol) {
				t.Errorf("Manhattan minSamples=%d: core[%d] = %v, want %v", minSamples, i, got[i], expected[i])
			}
		}
	}
}

func TestComputeCoreDistancesTree_MinSamples0(t *testing.T) {
	data := []float64{0, 0, 1, 1}
	tree := NewKDTree(data, 2, 2, EuclideanMetric{}, 10)

	core := ComputeCoreDistancesTree(tree, 0)
	for i := 0; i < 2; i++ {
		if core[i] != 0 {
			t.Errorf("core[%d] = %v, want 0 for minSamples=0", i, core[i])
		}
	}
}

func TestComputeCoreDistancesTree_EmptyData(t *testing.T) {
	tree := NewKDTree(nil, 0, 2, EuclideanMetric{}, 10)
	core := ComputeCoreDistancesTree(tree, 5)
	if core != nil {
		t.Errorf("expected nil for empty tree, got %v", core)
	}
}

func TestComputeCoreDistancesTree_LargerGrid(t *testing.T) {
	// 5x5 grid = 25 points in 2D
	n, dims := 25, 2
	data := make([]float64, n*dims)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			idx := i*5 + j
			data[idx*dims] = float64(i)
			data[idx*dims+1] = float64(j)
		}
	}
	metric := EuclideanMetric{}
	distMatrix := ComputePairwiseDistances(data, n, dims, metric)

	for minSamples := 1; minSamples <= 5; minSamples++ {
		expected := ComputeCoreDistances(distMatrix, n, minSamples)

		for _, builder := range []struct {
			name string
			tree SpatialTree
		}{
			{"KDTree", NewKDTree(data, n, dims, metric, 3)},
			{"BallTree", NewBallTree(data, n, dims, metric, 3)},
		} {
			got := ComputeCoreDistancesTree(builder.tree, minSamples)
			for i := 0; i < n; i++ {
				if !almostEqual(got[i], expected[i], floatTol) {
					t.Errorf("%s minSamples=%d: core[%d] = %v, want %v",
						builder.name, minSamples, i, got[i], expected[i])
				}
			}
		}
	}
}
