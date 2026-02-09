package hdbscan

import (
	"math"
	"testing"
)

func TestComputeCoreDistances_3Points_MinSamples1(t *testing.T) {
	// Points: (0,0), (3,0), (0,4) -- distances: d01=3, d02=4, d12=5
	distMatrix := []float64{
		0, 3, 4,
		3, 0, 5,
		4, 5, 0,
	}
	n := 3

	// minSamples=1: nearest-neighbor distance for each point
	core := ComputeCoreDistances(distMatrix, n, 1)
	expected := []float64{3, 3, 4}

	for i := 0; i < n; i++ {
		if !almostEqual(core[i], expected[i], floatTol) {
			t.Errorf("core[%d] = %v, expected %v", i, core[i], expected[i])
		}
	}
}

func TestComputeCoreDistances_3Points_MinSamples2(t *testing.T) {
	distMatrix := []float64{
		0, 3, 4,
		3, 0, 5,
		4, 5, 0,
	}
	n := 3

	// minSamples=2: 2nd nearest neighbor (max of non-self distances for 3 points)
	core := ComputeCoreDistances(distMatrix, n, 2)
	expected := []float64{4, 5, 5}

	for i := 0; i < n; i++ {
		if !almostEqual(core[i], expected[i], floatTol) {
			t.Errorf("core[%d] = %v, expected %v", i, core[i], expected[i])
		}
	}
}

func TestComputeCoreDistances_MinSamplesGE_N(t *testing.T) {
	distMatrix := []float64{
		0, 3, 4,
		3, 0, 5,
		4, 5, 0,
	}
	n := 3

	// minSamples=5 > n: should clamp to n-1=2
	core := ComputeCoreDistances(distMatrix, n, 5)
	expected := []float64{4, 5, 5}

	for i := 0; i < n; i++ {
		if !almostEqual(core[i], expected[i], floatTol) {
			t.Errorf("core[%d] = %v, expected %v", i, core[i], expected[i])
		}
	}
}

func TestComputeCoreDistances_MinSamplesN_Minus1(t *testing.T) {
	distMatrix := []float64{
		0, 3, 4,
		3, 0, 5,
		4, 5, 0,
	}
	n := 3

	// minSamples=n-1=2: should return max distance in each row (excluding self)
	core := ComputeCoreDistances(distMatrix, n, 2)
	expected := []float64{4, 5, 5}

	for i := 0; i < n; i++ {
		if !almostEqual(core[i], expected[i], floatTol) {
			t.Errorf("core[%d] = %v, expected %v", i, core[i], expected[i])
		}
	}
}

func TestComputeCoreDistances_5Points(t *testing.T) {
	// 5 points in 2D on x-axis: (0,0), (1,0), (2,0), (3,0), (4,0)
	data := []float64{0, 0, 1, 0, 2, 0, 3, 0, 4, 0}
	n, dims := 5, 2
	distMatrix := ComputePairwiseDistances(data, n, dims, EuclideanMetric{})

	// minSamples=2: 2nd nearest neighbor
	core := ComputeCoreDistances(distMatrix, n, 2)
	expected := []float64{2, 1, 1, 1, 2}

	for i := 0; i < n; i++ {
		if !almostEqual(core[i], expected[i], floatTol) {
			t.Errorf("core[%d] = %v, expected %v", i, core[i], expected[i])
		}
	}
}

func TestComputeCoreDistances_SinglePoint(t *testing.T) {
	// n=1: only one point, minSamples clamped to 0
	distMatrix := []float64{0}
	core := ComputeCoreDistances(distMatrix, 1, 5)
	if len(core) != 1 {
		t.Fatalf("expected length 1, got %d", len(core))
	}
	if core[0] != 0 {
		t.Errorf("expected 0 for single point, got %v", core[0])
	}
}

func TestComputeCoreDistances_IdenticalPoints(t *testing.T) {
	distMatrix := []float64{
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
	}
	n := 3
	core := ComputeCoreDistances(distMatrix, n, 1)
	for i := 0; i < n; i++ {
		if core[i] != 0 {
			t.Errorf("core[%d] = %v, expected 0 for identical points", i, core[i])
		}
	}
}

func TestComputeCoreDistances_InfDistances(t *testing.T) {
	distMatrix := []float64{
		0, math.Inf(1), 2,
		math.Inf(1), 0, 3,
		2, 3, 0,
	}
	n := 3
	core := ComputeCoreDistances(distMatrix, n, 1)
	expected := []float64{2, 3, 2}
	for i := 0; i < n; i++ {
		if !almostEqual(core[i], expected[i], floatTol) {
			t.Errorf("core[%d] = %v, expected %v", i, core[i], expected[i])
		}
	}
}
