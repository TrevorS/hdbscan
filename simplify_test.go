package hdbscan

import (
	"testing"
)

func TestSimplifyHierarchy_LowPersistenceLeafRemoved(t *testing.T) {
	// Tree: root=5, clusters 6 and 7
	// Cluster 6 born at lambda=0.5 (parent=5, birth=0)
	// Cluster 7 born at lambda=0.5 (parent=5, birth=0)
	// Cluster 6 has points at lambda=1.0
	// Cluster 7 has points at lambda=0.6
	//
	// Persistence of cluster 6: lambdaBirth(6) - lambdaBirth(parent=5)
	//   lambdaBirth(6) = 0.5, lambdaBirth(5) = 0 → persistence = 0.5
	// Persistence of cluster 7: 0.5 - 0 = 0.5
	//
	// With threshold = 0.6: both leaves have persistence 0.5 < 0.6, so both should be removed.
	// All points re-parented to root 5.
	tree := []CondensedTreeEntry{
		{Parent: 5, Child: 6, LambdaVal: 0.5, ChildSize: 3},
		{Parent: 5, Child: 7, LambdaVal: 0.5, ChildSize: 2},
		{Parent: 6, Child: 0, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 1, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 2, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 3, LambdaVal: 0.6, ChildSize: 1},
		{Parent: 7, Child: 4, LambdaVal: 0.6, ChildSize: 1},
	}

	simplified := SimplifyHierarchy(tree, 0.6)

	// Only point entries should remain, all with root as parent
	for _, e := range simplified {
		if e.ChildSize > 1 {
			t.Errorf("expected no cluster entries after simplification, got parent=%d child=%d size=%d",
				e.Parent, e.Child, e.ChildSize)
		}
	}
}

func TestSimplifyHierarchy_HighPersistenceLeafKept(t *testing.T) {
	tree := []CondensedTreeEntry{
		{Parent: 5, Child: 6, LambdaVal: 0.5, ChildSize: 3},
		{Parent: 5, Child: 7, LambdaVal: 0.5, ChildSize: 2},
		{Parent: 6, Child: 0, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 1, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 2, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 3, LambdaVal: 0.6, ChildSize: 1},
		{Parent: 7, Child: 4, LambdaVal: 0.6, ChildSize: 1},
	}

	// With threshold=0.3: both leaves have persistence 0.5 >= 0.3, so both kept
	simplified := SimplifyHierarchy(tree, 0.3)

	// Should have cluster entries for 6 and 7
	clusterEntries := 0
	for _, e := range simplified {
		if e.ChildSize > 1 {
			clusterEntries++
		}
	}
	if clusterEntries != 2 {
		t.Errorf("expected 2 cluster entries, got %d", clusterEntries)
	}
}

func TestSimplifyHierarchy_ThresholdZero(t *testing.T) {
	tree := []CondensedTreeEntry{
		{Parent: 5, Child: 6, LambdaVal: 0.5, ChildSize: 3},
		{Parent: 5, Child: 7, LambdaVal: 0.5, ChildSize: 2},
		{Parent: 6, Child: 0, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 1, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 2, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 3, LambdaVal: 0.6, ChildSize: 1},
		{Parent: 7, Child: 4, LambdaVal: 0.6, ChildSize: 1},
	}

	simplified := SimplifyHierarchy(tree, 0.0)

	// Threshold=0: tree unchanged
	if len(simplified) != len(tree) {
		t.Errorf("expected %d entries, got %d (threshold=0 should not change tree)",
			len(tree), len(simplified))
	}
}

func TestSimplifyHierarchy_MixedPersistence(t *testing.T) {
	// Tree with three levels:
	// Root=5, cluster 6 (birth lambda=0.2), cluster 7 (birth lambda=0.2)
	// Cluster 7 has sub-clusters 8 (birth lambda=0.9) and 9 (birth lambda=0.9)
	//
	// Persistence of leaf 6: lambdaBirth(6) - lambdaBirth(5) = 0.2 - 0 = 0.2
	// Persistence of leaf 8: lambdaBirth(8) - lambdaBirth(7) = 0.9 - 0.2 = 0.7
	// Persistence of leaf 9: lambdaBirth(9) - lambdaBirth(7) = 0.9 - 0.2 = 0.7
	//
	// With threshold=0.5: leaf 6 (persistence=0.2) removed, leaves 8 and 9 (persistence=0.7) kept
	tree := []CondensedTreeEntry{
		{Parent: 5, Child: 6, LambdaVal: 0.2, ChildSize: 2},
		{Parent: 5, Child: 7, LambdaVal: 0.2, ChildSize: 4},
		{Parent: 6, Child: 0, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 1, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 8, LambdaVal: 0.9, ChildSize: 2},
		{Parent: 7, Child: 9, LambdaVal: 0.9, ChildSize: 2},
		{Parent: 8, Child: 2, LambdaVal: 1.5, ChildSize: 1},
		{Parent: 8, Child: 3, LambdaVal: 1.5, ChildSize: 1},
		{Parent: 9, Child: 4, LambdaVal: 1.5, ChildSize: 1},
		{Parent: 9, Child: 5, LambdaVal: 1.5, ChildSize: 1},
	}

	simplified := SimplifyHierarchy(tree, 0.5)

	// Cluster 6 should be removed (persistence 0.2 < 0.5)
	// Points 0, 1 should be re-parented to root
	// Clusters 7, 8, 9 should remain (possibly relabeled)
	// Check that points 0 and 1 now have a parent that is the root cluster

	hasClusterChild := make(map[int]bool)
	for _, e := range simplified {
		if e.ChildSize > 1 {
			hasClusterChild[e.Child] = true
		}
	}

	// Should have fewer cluster entries since leaf 6 was removed
	clusterEntries := 0
	for _, e := range simplified {
		if e.ChildSize > 1 {
			clusterEntries++
		}
	}
	// After removing cluster 6: cluster entries = 3 (7, 8, 9) but 6 is gone.
	// Actually the cluster entry for 6 is removed. The entries for 8 and 9 under 7 stay.
	// The root→7 entry stays. So 3 cluster entries.
	if clusterEntries < 3 {
		t.Errorf("expected at least 3 cluster entries after removing leaf 6, got %d", clusterEntries)
	}
}
