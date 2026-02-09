package hdbscan

import (
	"math"
	"testing"
)

// helper: sum all MST edge weights.
func totalMSTWeight(edges [][3]float64) float64 {
	total := 0.0
	for _, e := range edges {
		total += e[2]
	}
	return total
}

// helper: build a flat n×n row-major matrix from a 2D slice.
func flatMatrix(m [][]float64) []float64 {
	n := len(m)
	flat := make([]float64, n*n)
	for i := range m {
		for j := range m[i] {
			flat[i*n+j] = m[i][j]
		}
	}
	return flat
}

func TestPrimMST_FourPointKnownMST(t *testing.T) {
	// Distance matrix:
	//      0  1  3  4
	//      1  0  2  5
	//      3  2  0  1
	//      4  5  1  0
	// Known MST edges (by weight): {0,1}=1, {2,3}=1, {1,2}=2  total=4
	dist := flatMatrix([][]float64{
		{0, 1, 3, 4},
		{1, 0, 2, 5},
		{3, 2, 0, 1},
		{4, 5, 1, 0},
	})

	edges := PrimMST(dist, 4)

	if len(edges) != 3 {
		t.Fatalf("expected 3 edges, got %d", len(edges))
	}

	total := totalMSTWeight(edges)
	if math.Abs(total-4.0) > 1e-10 {
		t.Errorf("expected total MST weight 4.0, got %f", total)
	}

	// Verify individual edge weights are {1, 1, 2} in some order.
	weights := make(map[float64]int)
	for _, e := range edges {
		weights[e[2]]++
	}
	if weights[1.0] != 2 || weights[2.0] != 1 {
		t.Errorf("expected weights {1:2, 2:1}, got %v", weights)
	}
}

func TestPrimMST_InfEdges(t *testing.T) {
	// 3-point graph where one distance is +Inf.
	// 0-1: 2, 0-2: +Inf, 1-2: 3
	// MST: {0,1}=2, {1,2}=3
	inf := math.Inf(1)
	dist := flatMatrix([][]float64{
		{0, 2, inf},
		{2, 0, 3},
		{inf, 3, 0},
	})

	edges := PrimMST(dist, 3)

	if len(edges) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(edges))
	}

	total := totalMSTWeight(edges)
	if math.Abs(total-5.0) > 1e-10 {
		t.Errorf("expected total MST weight 5.0, got %f", total)
	}
}

func TestPrimMST_InfEdgeInMST(t *testing.T) {
	// 3-point graph where MST must include +Inf edge.
	// 0-1: 2, 0-2: +Inf, 1-2: +Inf
	// MST: {0,1}=2, then one of the +Inf edges (no other choice)
	inf := math.Inf(1)
	dist := flatMatrix([][]float64{
		{0, 2, inf},
		{2, 0, inf},
		{inf, inf, 0},
	})

	edges := PrimMST(dist, 3)

	if len(edges) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(edges))
	}

	// One edge should be +Inf.
	hasInf := false
	for _, e := range edges {
		if math.IsInf(e[2], 1) {
			hasInf = true
		}
	}
	if !hasInf {
		t.Error("expected at least one +Inf edge in MST")
	}
}

func TestPrimMST_SinglePoint(t *testing.T) {
	dist := []float64{0}
	edges := PrimMST(dist, 1)

	if len(edges) != 0 {
		t.Fatalf("expected 0 edges for n=1, got %d", len(edges))
	}
}

func TestPrimMST_TwoPoints(t *testing.T) {
	dist := flatMatrix([][]float64{
		{0, 5},
		{5, 0},
	})

	edges := PrimMST(dist, 2)

	if len(edges) != 1 {
		t.Fatalf("expected 1 edge for n=2, got %d", len(edges))
	}
	if math.Abs(edges[0][2]-5.0) > 1e-10 {
		t.Errorf("expected edge weight 5.0, got %f", edges[0][2])
	}
}

func TestPrimMST_SixPoint(t *testing.T) {
	// 6-point complete graph (symmetric).
	// Hand-computed from a simple structure:
	//   0-1:1, 0-2:4, 0-3:7, 0-4:10, 0-5:13
	//   1-2:2, 1-3:6, 1-4:9,  1-5:12
	//   2-3:3, 2-4:8, 2-5:11
	//   3-4:5, 3-5:10
	//   4-5:6
	// MST (greedy): {0,1}=1, {1,2}=2, {2,3}=3, {3,4}=5, {4,5}=6  total=17
	dist := flatMatrix([][]float64{
		{0, 1, 4, 7, 10, 13},
		{1, 0, 2, 6, 9, 12},
		{4, 2, 0, 3, 8, 11},
		{7, 6, 3, 0, 5, 10},
		{10, 9, 8, 5, 0, 6},
		{13, 12, 11, 10, 6, 0},
	})

	edges := PrimMST(dist, 6)

	if len(edges) != 5 {
		t.Fatalf("expected 5 edges, got %d", len(edges))
	}

	total := totalMSTWeight(edges)
	if math.Abs(total-17.0) > 1e-10 {
		t.Errorf("expected total MST weight 17.0, got %f", total)
	}
}
