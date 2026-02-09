package hdbscan

import (
	"math"
	"testing"
)

func TestMutualReachability_3Points_Alpha1(t *testing.T) {
	// 3 points: (0,0), (3,0), (0,4) — distances: d01=3, d02=4, d12=5
	distMatrix := []float64{
		0, 3, 4,
		3, 0, 5,
		4, 5, 0,
	}
	// Core distances with minSamples=1: [3, 3, 4]
	coreDistances := []float64{3, 3, 4}
	n := 3

	mr := MutualReachability(distMatrix, coreDistances, n, 1.0)

	// mr[i][j] = max(core[i], core[j], dist[i][j])
	//
	// mr[0][0] = max(3, 3, 0) = 3   (diagonal = max of core distances)
	// mr[0][1] = max(3, 3, 3) = 3
	// mr[0][2] = max(3, 4, 4) = 4
	// mr[1][0] = max(3, 3, 3) = 3
	// mr[1][1] = max(3, 3, 0) = 3
	// mr[1][2] = max(3, 4, 5) = 5
	// mr[2][0] = max(4, 3, 4) = 4
	// mr[2][1] = max(4, 3, 5) = 5
	// mr[2][2] = max(4, 4, 0) = 4
	expected := []float64{
		3, 3, 4,
		3, 3, 5,
		4, 5, 4,
	}

	for i := 0; i < n*n; i++ {
		if !almostEqual(mr[i], expected[i], floatTol) {
			row, col := i/n, i%n
			t.Errorf("mr[%d,%d] = %v, expected %v", row, col, mr[i], expected[i])
		}
	}
}

func TestMutualReachability_Symmetry(t *testing.T) {
	distMatrix := []float64{
		0, 3, 4,
		3, 0, 5,
		4, 5, 0,
	}
	coreDistances := []float64{3, 3, 4}
	n := 3

	mr := MutualReachability(distMatrix, coreDistances, n, 1.0)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !almostEqual(mr[i*n+j], mr[j*n+i], floatTol) {
				t.Errorf("mr[%d,%d]=%v != mr[%d,%d]=%v", i, j, mr[i*n+j], j, i, mr[j*n+i])
			}
		}
	}
}

func TestMutualReachability_Alpha05(t *testing.T) {
	// alpha=0.5: raw distances doubled before max
	distMatrix := []float64{
		0, 3, 4,
		3, 0, 5,
		4, 5, 0,
	}
	coreDistances := []float64{3, 3, 4}
	n := 3

	mr := MutualReachability(distMatrix, coreDistances, n, 0.5)

	// d/alpha doubles the raw distances:
	// mr[0][0] = max(3, 3, 0/0.5=0) = 3
	// mr[0][1] = max(3, 3, 3/0.5=6) = 6
	// mr[0][2] = max(3, 4, 4/0.5=8) = 8
	// mr[1][0] = max(3, 3, 3/0.5=6) = 6
	// mr[1][1] = max(3, 3, 0/0.5=0) = 3
	// mr[1][2] = max(3, 4, 5/0.5=10) = 10
	// mr[2][0] = max(4, 3, 4/0.5=8) = 8
	// mr[2][1] = max(4, 3, 5/0.5=10) = 10
	// mr[2][2] = max(4, 4, 0/0.5=0) = 4
	expected := []float64{
		3, 6, 8,
		6, 3, 10,
		8, 10, 4,
	}

	for i := 0; i < n*n; i++ {
		if !almostEqual(mr[i], expected[i], floatTol) {
			row, col := i/n, i%n
			t.Errorf("mr[%d,%d] = %v, expected %v", row, col, mr[i], expected[i])
		}
	}
}

func TestMutualReachability_Alpha1_MatchesDivisionPath(t *testing.T) {
	// Verify alpha=1.0 fast path matches what would happen with division
	distMatrix := []float64{
		0, 3, 4,
		3, 0, 5,
		4, 5, 0,
	}
	coreDistances := []float64{3, 3, 4}
	n := 3

	mr1 := MutualReachability(distMatrix, coreDistances, n, 1.0)

	// Compute manually with division: d/1.0 == d
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			d := distMatrix[i*n+j] / 1.0
			expected := d
			if coreDistances[i] > expected {
				expected = coreDistances[i]
			}
			if coreDistances[j] > expected {
				expected = coreDistances[j]
			}
			if !almostEqual(mr1[i*n+j], expected, floatTol) {
				t.Errorf("mr[%d,%d] = %v, manual = %v", i, j, mr1[i*n+j], expected)
			}
		}
	}
}

func TestMutualReachability_Diagonal(t *testing.T) {
	// Diagonal should be max(core[i], core[i], 0) = core[i]
	distMatrix := []float64{
		0, 3, 4,
		3, 0, 5,
		4, 5, 0,
	}
	coreDistances := []float64{3, 3, 4}
	n := 3

	mr := MutualReachability(distMatrix, coreDistances, n, 1.0)

	for i := 0; i < n; i++ {
		if !almostEqual(mr[i*n+i], coreDistances[i], floatTol) {
			t.Errorf("diagonal mr[%d,%d] = %v, expected core[%d] = %v",
				i, i, mr[i*n+i], i, coreDistances[i])
		}
	}
}

func TestMutualReachability_LargeCoreDistances(t *testing.T) {
	// When core distances dominate, MR should reflect that
	distMatrix := []float64{
		0, 1,
		1, 0,
	}
	coreDistances := []float64{10, 20}
	n := 2

	mr := MutualReachability(distMatrix, coreDistances, n, 1.0)

	// mr[0][0] = max(10, 10, 0) = 10
	// mr[0][1] = max(10, 20, 1) = 20
	// mr[1][0] = max(20, 10, 1) = 20
	// mr[1][1] = max(20, 20, 0) = 20
	expected := []float64{10, 20, 20, 20}

	for i := 0; i < 4; i++ {
		if !almostEqual(mr[i], expected[i], floatTol) {
			t.Errorf("mr[%d] = %v, expected %v", i, mr[i], expected[i])
		}
	}
}

func TestMutualReachability_InfDistances(t *testing.T) {
	distMatrix := []float64{
		0, math.Inf(1),
		math.Inf(1), 0,
	}
	coreDistances := []float64{math.Inf(1), math.Inf(1)}
	n := 2

	mr := MutualReachability(distMatrix, coreDistances, n, 1.0)

	// All entries should be +Inf
	for i := 0; i < 4; i++ {
		if !math.IsInf(mr[i], 1) {
			t.Errorf("mr[%d] = %v, expected +Inf", i, mr[i])
		}
	}
}
