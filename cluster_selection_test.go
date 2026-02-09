package hdbscan

import (
	"math"
	"testing"
)

// sixPointTree returns the condensed tree and stability for the 6-point example
// with minClusterSize=2. Clusters: 6(root), 7, 8, 9, 10.
func sixPointTree() ([]CondensedTreeEntry, map[int]float64) {
	tree := []CondensedTreeEntry{
		{Parent: 6, Child: 7, LambdaVal: 0.2, ChildSize: 2},
		{Parent: 6, Child: 8, LambdaVal: 0.2, ChildSize: 4},
		{Parent: 8, Child: 9, LambdaVal: 1.0 / 3.0, ChildSize: 2},
		{Parent: 8, Child: 10, LambdaVal: 1.0 / 3.0, ChildSize: 2},
		{Parent: 7, Child: 4, LambdaVal: 0.5, ChildSize: 1},
		{Parent: 7, Child: 5, LambdaVal: 0.5, ChildSize: 1},
		{Parent: 9, Child: 0, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 9, Child: 1, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 10, Child: 2, LambdaVal: 1.0 / 1.5, ChildSize: 1},
		{Parent: 10, Child: 3, LambdaVal: 1.0 / 1.5, ChildSize: 1},
	}
	stab := ComputeStability(tree)
	return tree, stab
}

func TestSelectClustersEOM_TwoCluster(t *testing.T) {
	// Simple tree: root cluster with two leaf children
	tree := []CondensedTreeEntry{
		{Parent: 5, Child: 6, LambdaVal: 0.5, ChildSize: 3},
		{Parent: 5, Child: 7, LambdaVal: 0.5, ChildSize: 2},
		{Parent: 6, Child: 0, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 1, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 2, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 3, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 4, LambdaVal: 1.0, ChildSize: 1},
	}
	stab := ComputeStability(tree)

	// allowSingleCluster=false: root excluded from candidates
	selected, _ := SelectClustersEOM(tree, stab, false, 0, math.Inf(1))

	// Leaves 6 and 7 should be selected
	if !selected[6] || !selected[7] {
		t.Errorf("expected clusters 6 and 7 selected, got %v", selected)
	}
	if selected[5] {
		t.Error("root cluster 5 should not be selected")
	}
}

func TestSelectClustersEOM_ParentVsChildren(t *testing.T) {
	tree, stab := sixPointTree()

	// stabilities: 6=1.2, 7=0.6, 8=8/15≈0.533, 9=4/3≈1.333, 10=2/3≈0.667
	//
	// Leaf clusters: 7, 9, 10 (no cluster children)
	// Non-leaf clusters (excluding root): 8
	//
	// Bottom-up from 8:
	//   Children stability sum = stability[9] + stability[10] = 4/3 + 2/3 = 2.0
	//   stability[8] = 8/15 ≈ 0.533 < 2.0
	//   → children win, stability[8] updated to 2.0
	//
	// So selected = {7, 9, 10}

	selected, updatedStab := SelectClustersEOM(tree, stab, false, 0, math.Inf(1))

	if !selected[7] || !selected[9] || !selected[10] {
		t.Errorf("expected 7, 9, 10 selected, got %v", selected)
	}
	if selected[8] {
		t.Error("cluster 8 should not be selected (children won)")
	}
	if selected[6] {
		t.Error("root cluster 6 should not be selected")
	}

	// Updated stability for 8 should be children's sum
	assertFloat(t, "updatedStab[8]", updatedStab[8], 2.0, 1e-10)
}

func TestSelectClustersEOM_MaxClusterSize(t *testing.T) {
	// Two-cluster tree where one cluster is large
	tree := []CondensedTreeEntry{
		{Parent: 5, Child: 6, LambdaVal: 0.5, ChildSize: 3},
		{Parent: 5, Child: 7, LambdaVal: 0.5, ChildSize: 2},
		{Parent: 6, Child: 0, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 1, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 2, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 3, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 4, LambdaVal: 1.0, ChildSize: 1},
	}
	stab := ComputeStability(tree)

	// Max cluster size = 2 should not include cluster 6 (size=3)
	// But 6 is a leaf with no cluster children, so maxClusterSize forces it
	// to prefer children — but it has no cluster children!
	// Actually the reference just checks > maxClusterSize and if so, children win.
	// But if there are no cluster children, the cluster is still selected.
	// maxClusterSize primarily affects non-leaf clusters.
	selected, _ := SelectClustersEOM(tree, stab, false, 2, math.Inf(1))
	if !selected[6] || !selected[7] {
		t.Errorf("leaf clusters should be selected regardless of maxClusterSize, got %v", selected)
	}
}

func TestSelectClustersEOM_AllowSingleCluster(t *testing.T) {
	// Tree where root cluster's stability > children
	tree := []CondensedTreeEntry{
		{Parent: 3, Child: 4, LambdaVal: 0.1, ChildSize: 2},
		{Parent: 3, Child: 5, LambdaVal: 0.1, ChildSize: 1},
		{Parent: 4, Child: 0, LambdaVal: 0.2, ChildSize: 1},
		{Parent: 4, Child: 1, LambdaVal: 0.2, ChildSize: 1},
		{Parent: 3, Child: 2, LambdaVal: 1.0, ChildSize: 1},
	}
	stab := ComputeStability(tree)

	// With allowSingleCluster=true, root is included in candidates
	selected, _ := SelectClustersEOM(tree, stab, true, 0, math.Inf(1))

	// Root stability should be compared against children
	// stab[3] = root, stab[4] = child cluster, stab[5] = child cluster
	// If root wins, all descendants deselected
	_ = selected // Just ensure no panic for now
}

// Tests for epsilon search and leaf selection

func TestSelectClustersLeaf_CorrectLeaves(t *testing.T) {
	tree, _ := sixPointTree()

	// Leaf clusters are those with no cluster children: 7, 9, 10
	selected := SelectClustersLeaf(tree, 0.0)

	if !selected[7] || !selected[9] || !selected[10] {
		t.Errorf("expected leaves 7, 9, 10, got %v", selected)
	}
	if selected[6] || selected[8] {
		t.Error("non-leaf clusters should not be selected")
	}
}

func TestEpsilonSearch_BelowThreshold(t *testing.T) {
	tree, _ := sixPointTree()

	// Cluster 9 has lambdaBirth = 1/3, epsilon = 3.0
	// Cluster 10 has lambdaBirth = 1/3, epsilon = 3.0
	// Cluster 7 has lambdaBirth = 0.2, epsilon = 5.0
	//
	// With epsilon threshold = 4.0:
	//   Cluster 9 (eps=3.0 < 4.0) → traverse up to parent 8 (eps=1/0.2=5.0 >= 4.0) → 8
	//   Cluster 10 (eps=3.0 < 4.0) → traverse up to parent 8 (eps=5.0 >= 4.0) → 8
	//   Cluster 7 (eps=5.0 >= 4.0) → keep
	// Result: {7, 8}
	candidates := map[int]bool{7: true, 9: true, 10: true}
	result := EpsilonSearch(tree, candidates, 4.0, false)

	if !result[7] || !result[8] {
		t.Errorf("expected {7, 8}, got %v", result)
	}
	if result[9] || result[10] {
		t.Error("clusters 9 and 10 should have been merged upward")
	}
}

func TestEpsilonSearch_Deduplication(t *testing.T) {
	tree, _ := sixPointTree()

	// Both 9 and 10 should merge to 8, producing only one entry for 8
	candidates := map[int]bool{9: true, 10: true}
	result := EpsilonSearch(tree, candidates, 4.0, false)

	if !result[8] {
		t.Errorf("expected cluster 8 in result, got %v", result)
	}
	if result[9] || result[10] {
		t.Error("clusters 9 and 10 should have been merged to 8")
	}
}

func TestSelectClustersLeaf_WithEpsilon(t *testing.T) {
	tree, _ := sixPointTree()

	// Leaf clusters: 7, 9, 10
	// With epsilon=4.0: 9 and 10 merge to 8
	selected := SelectClustersLeaf(tree, 4.0)

	if !selected[7] || !selected[8] {
		t.Errorf("expected {7, 8}, got %v", selected)
	}
	if selected[9] || selected[10] {
		t.Error("9 and 10 should have been merged to 8")
	}
}
