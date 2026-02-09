package hdbscan

import (
	"math"
	"testing"
)

func TestLabel_FourPointMST(t *testing.T) {
	// 4 points. MST edges sorted by weight:
	//   [0,2, 1.0], [2,3, 1.0], [0,1, 2.0]
	//
	// Hand-traced through the reference UnionFind (2N-1 elements, next_label starts at N):
	//
	// Step 0: edge [0,2,1.0]
	//   find(0)=0, find(2)=2
	//   output: [0, 2, 1.0, 2]
	//   union(0,2) → new cluster 4
	//
	// Step 1: edge [2,3,1.0]
	//   find(2)=4 (parent[2]=4), find(3)=3
	//   output: [4, 3, 1.0, 3]
	//   union(4,3) → new cluster 5
	//
	// Step 2: edge [0,1,2.0]
	//   find(0)=5 (0→4→5), find(1)=1
	//   output: [5, 1, 2.0, 4]
	//   union(5,1) → new cluster 6
	edges := [][3]float64{
		{0, 2, 1.0},
		{2, 3, 1.0},
		{0, 1, 2.0},
	}

	dendro := Label(edges, 4)

	if len(dendro) != 3 {
		t.Fatalf("expected 3 dendrogram rows, got %d", len(dendro))
	}

	expected := [][4]float64{
		{0, 2, 1.0, 2},
		{4, 3, 1.0, 3},
		{5, 1, 2.0, 4},
	}

	for i, row := range dendro {
		for j := 0; j < 4; j++ {
			if math.Abs(row[j]-expected[i][j]) > 1e-10 {
				t.Errorf("row[%d][%d] = %f, want %f", i, j, row[j], expected[i][j])
			}
		}
	}
}

func TestLabel_SinglePoint(t *testing.T) {
	edges := [][3]float64{}
	dendro := Label(edges, 1)

	if len(dendro) != 0 {
		t.Fatalf("expected 0 dendrogram rows for n=1, got %d", len(dendro))
	}
}

func TestLabel_TwoPoints(t *testing.T) {
	edges := [][3]float64{
		{0, 1, 3.5},
	}

	dendro := Label(edges, 2)

	if len(dendro) != 1 {
		t.Fatalf("expected 1 dendrogram row, got %d", len(dendro))
	}

	// find(0)=0, find(1)=1
	// output: [0, 1, 3.5, 2]
	row := dendro[0]
	if row[0] != 0 || row[1] != 1 {
		t.Errorf("expected cluster IDs [0,1], got [%f,%f]", row[0], row[1])
	}
	if math.Abs(row[2]-3.5) > 1e-10 {
		t.Errorf("expected distance 3.5, got %f", row[2])
	}
	if row[3] != 2 {
		t.Errorf("expected merged size 2, got %f", row[3])
	}
}

func TestLabel_SortsEdgesByWeight(t *testing.T) {
	// Provide edges in descending order — Label should sort them ascending.
	edges := [][3]float64{
		{0, 1, 5.0},
		{1, 2, 1.0},
	}

	dendro := Label(edges, 3)

	if len(dendro) != 2 {
		t.Fatalf("expected 2 dendrogram rows, got %d", len(dendro))
	}

	// After sort: [1,2,1.0] then [0,1,5.0]
	// Step 0: find(1)=1, find(2)=2 → [1, 2, 1.0, 2], union → cluster 3
	// Step 1: find(0)=0, find(1)=3 → [0, 3, 5.0, 3], union → cluster 4
	if math.Abs(dendro[0][2]-1.0) > 1e-10 {
		t.Errorf("first merge should be at distance 1.0, got %f", dendro[0][2])
	}
	if math.Abs(dendro[1][2]-5.0) > 1e-10 {
		t.Errorf("second merge should be at distance 5.0, got %f", dendro[1][2])
	}
	// Final row merges all 3 points.
	if dendro[1][3] != 3 {
		t.Errorf("final merged size should be 3, got %f", dendro[1][3])
	}
}

func TestLabel_MergedSizesConsistent(t *testing.T) {
	// 5-point chain MST: 0-1(1), 1-2(2), 2-3(3), 3-4(4)
	edges := [][3]float64{
		{0, 1, 1.0},
		{1, 2, 2.0},
		{2, 3, 3.0},
		{3, 4, 4.0},
	}

	dendro := Label(edges, 5)

	if len(dendro) != 4 {
		t.Fatalf("expected 4 dendrogram rows, got %d", len(dendro))
	}

	// Last row should merge all 5 points.
	if dendro[3][3] != 5 {
		t.Errorf("final merged size should be 5, got %f", dendro[3][3])
	}

	// Each step's merged size should be > previous.
	for i := 1; i < len(dendro); i++ {
		if dendro[i][3] <= dendro[i-1][3] {
			t.Errorf("merged size should increase: row %d (%f) <= row %d (%f)",
				i, dendro[i][3], i-1, dendro[i-1][3])
		}
	}
}
