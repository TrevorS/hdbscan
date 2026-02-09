package hdbscan

import (
	"math"
	"testing"
)

// buildSimpleDendrogram creates a 6-point dendrogram for testing.
//
// Points: 0,1,2,3,4,5 (n=6)
// Dendrogram (scipy format): [left, right, distance, size]
//
//	Row 0: merge(0, 1) at dist=1.0 → node 6, size=2
//	Row 1: merge(2, 3) at dist=1.5 → node 7, size=2
//	Row 2: merge(4, 5) at dist=2.0 → node 8, size=2
//	Row 3: merge(6, 7) at dist=3.0 → node 9, size=4
//	Row 4: merge(8, 9) at dist=5.0 → node 10, size=6
func buildSimpleDendrogram() [][4]float64 {
	return [][4]float64{
		{0, 1, 1.0, 2},
		{2, 3, 1.5, 2},
		{4, 5, 2.0, 2},
		{6, 7, 3.0, 4},
		{8, 9, 5.0, 6},
	}
}

func TestCondenseTree_SixPointMinClusterSize2(t *testing.T) {
	dend := buildSimpleDendrogram()
	tree := CondenseTree(dend, 2)

	if len(tree) == 0 {
		t.Fatal("expected non-empty condensed tree")
	}

	// With minClusterSize=2, all merges produce children of size >= 2.
	// The root is node 10 in the dendrogram (2*5=10).
	// The reference code relabels root to num_points (6), next_label starts at 7.
	//
	// BFS from root (node 10):
	//   node 10 merges node 8 (size=2) and node 9 (size=4) at dist=5.0, lambda=0.2
	//   Both >= 2, so two new clusters: relabel[8]=7, relabel[9]=8
	//   Entry: (parent=6, child=7, lambda=0.2, size=2)
	//   Entry: (parent=6, child=8, lambda=0.2, size=4)
	//
	//   node 9 merges node 6 (size=2) and node 7 (size=2) at dist=3.0, lambda=1/3
	//   Both >= 2, so: relabel[6]=9, relabel[7]=10
	//   Entry: (parent=8, child=9, lambda=1/3, size=2)
	//   Entry: (parent=8, child=10, lambda=1/3, size=2)
	//
	//   node 8 merges point 4 (size=1) and point 5 (size=1) at dist=2.0, lambda=0.5
	//   Both < 2, collapse both points into parent cluster 7
	//   Entry: (parent=7, child=4, lambda=0.5, size=1)
	//   Entry: (parent=7, child=5, lambda=0.5, size=1)
	//
	//   node 6 merges point 0 (size=1) and point 1 (size=1) at dist=1.0, lambda=1.0
	//   Both < 2, collapse into parent cluster 9
	//   Entry: (parent=9, child=0, lambda=1.0, size=1)
	//   Entry: (parent=9, child=1, lambda=1.0, size=1)
	//
	//   node 7 merges point 2 (size=1) and point 3 (size=1) at dist=1.5, lambda=2/3
	//   Both < 2, collapse into parent cluster 10
	//   Entry: (parent=10, child=2, lambda=2/3, size=1)
	//   Entry: (parent=10, child=3, lambda=2/3, size=1)

	// Total entries: 10
	if len(tree) != 10 {
		t.Fatalf("expected 10 entries, got %d", len(tree))
	}

	// Root cluster (6) splits into two cluster children
	assertEntryExists(t, tree, 6, 7, 0.2, 2)
	assertEntryExists(t, tree, 6, 8, 0.2, 4)
	// Cluster 8 splits into two cluster children
	assertEntryApprox(t, tree, 8, 9, 1.0/3.0, 2)
	assertEntryApprox(t, tree, 8, 10, 1.0/3.0, 2)
	// Cluster 7 has two point children
	assertEntryExists(t, tree, 7, 4, 0.5, 1)
	assertEntryExists(t, tree, 7, 5, 0.5, 1)
	// Cluster 9 has two point children
	assertEntryExists(t, tree, 9, 0, 1.0, 1)
	assertEntryExists(t, tree, 9, 1, 1.0, 1)
	// Cluster 10 has two point children
	assertEntryApprox(t, tree, 10, 2, 1.0/1.5, 1)
	assertEntryApprox(t, tree, 10, 3, 1.0/1.5, 1)
}

func TestCondenseTree_SixPointMinClusterSize3(t *testing.T) {
	dend := buildSimpleDendrogram()
	tree := CondenseTree(dend, 3)

	// With minClusterSize=3:
	// Root (node 10, relabel=6) merges node 8 (size=2) and node 9 (size=4) at dist=5.0
	//   node 8 size=2 < 3, so collapse its points into root cluster 6
	//   node 9 size=4 >= 3, continues as same cluster (relabel=6)
	//   Point entries from node 8 subtree: points 4, 5 at lambda=0.2
	//
	// node 9 (relabel=6) merges node 6 (size=2) and node 7 (size=2) at dist=3.0
	//   Both < 3, collapse all points into cluster 6
	//   Points from node 6 subtree: 0, 1 at lambda=1/3
	//   Points from node 7 subtree: 2, 3 at lambda=1/3

	// Total: 6 point entries (all points collapse into root)
	if len(tree) != 6 {
		t.Fatalf("expected 6 entries, got %d", len(tree))
	}

	// All entries should have parent=6 and size=1
	for _, e := range tree {
		if e.Parent != 6 {
			t.Errorf("expected parent=6, got parent=%d for child=%d", e.Parent, e.Child)
		}
		if e.ChildSize != 1 {
			t.Errorf("expected childSize=1, got %d for child=%d", e.ChildSize, e.Child)
		}
	}

	// Check lambda values
	assertEntryExists(t, tree, 6, 4, 0.2, 1)
	assertEntryExists(t, tree, 6, 5, 0.2, 1)
	assertEntryApprox(t, tree, 6, 0, 1.0/3.0, 1)
	assertEntryApprox(t, tree, 6, 1, 1.0/3.0, 1)
	assertEntryApprox(t, tree, 6, 2, 1.0/3.0, 1)
	assertEntryApprox(t, tree, 6, 3, 1.0/3.0, 1)
}

func TestCondenseTree_AllIdenticalPoints(t *testing.T) {
	// All distances are 0 → lambda = +Inf
	dend := [][4]float64{
		{0, 1, 0.0, 2},
		{2, 3, 0.0, 2},
		{4, 5, 0.0, 4},
	}
	// Should not panic
	tree := CondenseTree(dend, 2)

	// Verify all lambdas are +Inf
	for _, e := range tree {
		if !math.IsInf(e.LambdaVal, 1) {
			t.Errorf("expected +Inf lambda for zero-distance dendrogram, got %f", e.LambdaVal)
		}
	}
}

func TestCondenseTree_SinglePoint(t *testing.T) {
	// n=1: empty dendrogram
	var dend [][4]float64
	tree := CondenseTree(dend, 2)
	if len(tree) != 0 {
		t.Fatalf("expected empty condensed tree for single point, got %d entries", len(tree))
	}
}

func TestCondenseTree_TwoPoints(t *testing.T) {
	// n=2: one dendrogram row
	dend := [][4]float64{
		{0, 1, 2.0, 2},
	}
	tree := CondenseTree(dend, 2)

	// Root is node 2 (n + 0 = 2), relabeled to 2 (num_points=2).
	// Both children are points (size=1 each), both < minClusterSize=2.
	// Collapse both into root.
	if len(tree) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(tree))
	}
	assertEntryExists(t, tree, 2, 0, 0.5, 1)
	assertEntryExists(t, tree, 2, 1, 0.5, 1)
}

// assertEntryExists checks that an exact entry exists in the tree.
func assertEntryExists(t *testing.T, tree []CondensedTreeEntry, parent, child int, lambda float64, childSize int) {
	t.Helper()
	for _, e := range tree {
		if e.Parent == parent && e.Child == child && e.ChildSize == childSize && e.LambdaVal == lambda {
			return
		}
	}
	t.Errorf("entry not found: parent=%d child=%d lambda=%f size=%d", parent, child, lambda, childSize)
}

// assertEntryApprox checks with floating point tolerance.
func assertEntryApprox(t *testing.T, tree []CondensedTreeEntry, parent, child int, lambda float64, childSize int) {
	t.Helper()
	const eps = 1e-10
	for _, e := range tree {
		if e.Parent == parent && e.Child == child && e.ChildSize == childSize &&
			math.Abs(e.LambdaVal-lambda) < eps {
			return
		}
	}
	t.Errorf("entry not found (approx): parent=%d child=%d lambda=%f size=%d", parent, child, lambda, childSize)
}
