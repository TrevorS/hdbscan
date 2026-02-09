package hdbscan

import (
	"math"
	"testing"
)

func TestComputePairwiseDistancesParallel_BitwiseIdentical(t *testing.T) {
	data := []float64{
		0, 0,
		3, 0,
		0, 4,
		1, 1,
		5, 5,
	}
	n, dims := 5, 2
	metric := EuclideanMetric{}

	sequential := ComputePairwiseDistances(data, n, dims, metric)

	for _, workers := range []int{1, 2, 4} {
		parallel := ComputePairwiseDistancesParallel(data, n, dims, metric, workers)

		if len(parallel) != len(sequential) {
			t.Fatalf("workers=%d: length mismatch %d != %d", workers, len(parallel), len(sequential))
		}

		for i := range sequential {
			if parallel[i] != sequential[i] {
				t.Errorf("workers=%d: result[%d] = %v, expected %v (bitwise)",
					workers, i, parallel[i], sequential[i])
			}
		}
	}
}

func TestComputePairwiseDistancesParallel_Manhattan(t *testing.T) {
	data := []float64{
		0, 0,
		3, 4,
		6, 0,
		1, 1,
	}
	n, dims := 4, 2
	metric := ManhattanMetric{}

	sequential := ComputePairwiseDistances(data, n, dims, metric)
	parallel := ComputePairwiseDistancesParallel(data, n, dims, metric, 3)

	for i := range sequential {
		if parallel[i] != sequential[i] {
			t.Errorf("Manhattan parallel[%d] = %v, expected %v", i, parallel[i], sequential[i])
		}
	}
}

func TestComputePairwiseDistancesParallel_SinglePoint(t *testing.T) {
	data := []float64{1, 2}
	n, dims := 1, 2

	result := ComputePairwiseDistancesParallel(data, n, dims, EuclideanMetric{}, 4)

	if len(result) != 1 {
		t.Fatalf("expected length 1, got %d", len(result))
	}
	if result[0] != 0 {
		t.Errorf("expected 0, got %v", result[0])
	}
}

func TestComputePairwiseDistancesParallel_TwoPoints(t *testing.T) {
	data := []float64{0, 0, 3, 4}
	n, dims := 2, 2

	result := ComputePairwiseDistancesParallel(data, n, dims, EuclideanMetric{}, 2)

	if len(result) != 4 {
		t.Fatalf("expected length 4, got %d", len(result))
	}
	if !almostEqual(result[0*n+1], 5.0, floatTol) {
		t.Errorf("expected 5.0, got %v", result[0*n+1])
	}
	if !almostEqual(result[1*n+0], 5.0, floatTol) {
		t.Errorf("expected 5.0, got %v", result[1*n+0])
	}
}

func TestComputePairwiseDistancesParallel_Symmetry(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	n, dims := 5, 2

	result := ComputePairwiseDistancesParallel(data, n, dims, EuclideanMetric{}, 3)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if result[i*n+j] != result[j*n+i] {
				t.Errorf("asymmetric: dist[%d,%d]=%v != dist[%d,%d]=%v",
					i, j, result[i*n+j], j, i, result[j*n+i])
			}
		}
	}
}

func TestComputePairwiseDistancesParallel_DiagonalZero(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	n, dims := 3, 2

	result := ComputePairwiseDistancesParallel(data, n, dims, EuclideanMetric{}, 2)

	for i := 0; i < n; i++ {
		if result[i*n+i] != 0 {
			t.Errorf("diagonal dist[%d,%d] = %v, expected 0", i, i, result[i*n+i])
		}
	}
}

func TestComputePairwiseDistancesParallel_MoreWorkersThanRows(t *testing.T) {
	data := []float64{0, 0, 3, 4, 6, 0}
	n, dims := 3, 2

	sequential := ComputePairwiseDistances(data, n, dims, EuclideanMetric{})
	parallel := ComputePairwiseDistancesParallel(data, n, dims, EuclideanMetric{}, 10)

	for i := range sequential {
		if parallel[i] != sequential[i] {
			t.Errorf("parallel[%d] = %v, expected %v", i, parallel[i], sequential[i])
		}
	}
}

func TestComputePairwiseDistancesParallel_LargerDataset(t *testing.T) {
	// Generate a 20-point dataset to exercise multiple workers with real load.
	n, dims := 20, 3
	data := make([]float64, n*dims)
	for i := range data {
		data[i] = math.Sin(float64(i) * 0.7)
	}

	sequential := ComputePairwiseDistances(data, n, dims, EuclideanMetric{})

	for _, workers := range []int{2, 4, 7} {
		parallel := ComputePairwiseDistancesParallel(data, n, dims, EuclideanMetric{}, workers)

		for i := range sequential {
			if parallel[i] != sequential[i] {
				t.Errorf("workers=%d: parallel[%d] = %v, expected %v",
					workers, i, parallel[i], sequential[i])
			}
		}
	}
}
