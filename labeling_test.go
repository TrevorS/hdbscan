package hdbscan

import "testing"

func TestGetLabelsAndProbabilities_TwoClusters(t *testing.T) {
	// Tree: root=5, clusters 6 and 7
	tree := []CondensedTreeEntry{
		{Parent: 5, Child: 6, LambdaVal: 0.5, ChildSize: 3},
		{Parent: 5, Child: 7, LambdaVal: 0.5, ChildSize: 2},
		{Parent: 6, Child: 0, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 1, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 6, Child: 2, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 3, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 4, LambdaVal: 1.0, ChildSize: 1},
	}

	selected := map[int]bool{6: true, 7: true}
	labels, probs := GetLabelsAndProbabilities(tree, selected, 5, false, 0.0, false)

	// Points 0,1,2 → cluster 6 → label 0
	// Points 3,4 → cluster 7 → label 1
	expectedLabels := []int{0, 0, 0, 1, 1}
	for i, l := range expectedLabels {
		if labels[i] != l {
			t.Errorf("labels[%d] = %d, want %d", i, labels[i], l)
		}
	}

	// All points at maxLambda → prob = 1.0
	for i, p := range probs {
		if p != 1.0 {
			t.Errorf("probs[%d] = %f, want 1.0", i, p)
		}
	}
}

func TestGetLabelsAndProbabilities_NoisePoints(t *testing.T) {
	// Tree with nested clusters. Points that are in the root but not a selected cluster → noise.
	tree := []CondensedTreeEntry{
		{Parent: 4, Child: 5, LambdaVal: 0.2, ChildSize: 2},
		{Parent: 4, Child: 0, LambdaVal: 0.2, ChildSize: 1}, // point 0 falls out early
		{Parent: 4, Child: 1, LambdaVal: 0.3, ChildSize: 1}, // point 1 falls out early
		{Parent: 5, Child: 2, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 5, Child: 3, LambdaVal: 1.0, ChildSize: 1},
	}

	selected := map[int]bool{5: true}
	labels, probs := GetLabelsAndProbabilities(tree, selected, 4, false, 0.0, false)

	// Points 0, 1 → in root cluster 4 (not selected) → noise
	// Points 2, 3 → in cluster 5 (selected) → label 0
	if labels[0] != -1 || labels[1] != -1 {
		t.Errorf("expected noise for points 0,1: labels=%v", labels)
	}
	if labels[2] != 0 || labels[3] != 0 {
		t.Errorf("expected label 0 for points 2,3: labels=%v", labels)
	}

	// Noise points have probability 0
	if probs[0] != 0.0 || probs[1] != 0.0 {
		t.Errorf("expected prob 0 for noise points: probs=%v", probs)
	}
}

func TestGetLabelsAndProbabilities_AllowSingleCluster_True(t *testing.T) {
	// Only the root cluster is selected
	tree := []CondensedTreeEntry{
		{Parent: 3, Child: 0, LambdaVal: 0.5, ChildSize: 1},
		{Parent: 3, Child: 1, LambdaVal: 0.5, ChildSize: 1},
		{Parent: 3, Child: 2, LambdaVal: 0.5, ChildSize: 1},
	}

	selected := map[int]bool{3: true}
	labels, probs := GetLabelsAndProbabilities(tree, selected, 3, true, 0.0, false)

	// All points in the single cluster → label 0
	for i, l := range labels {
		if l != 0 {
			t.Errorf("labels[%d] = %d, want 0 (single cluster)", i, l)
		}
	}

	// All at same lambda → prob = 1.0
	for i, p := range probs {
		if p != 1.0 {
			t.Errorf("probs[%d] = %f, want 1.0", i, p)
		}
	}
}

func TestGetLabelsAndProbabilities_AllowSingleCluster_False(t *testing.T) {
	// Only the root cluster is selected, but allowSingleCluster=false
	tree := []CondensedTreeEntry{
		{Parent: 3, Child: 0, LambdaVal: 0.5, ChildSize: 1},
		{Parent: 3, Child: 1, LambdaVal: 0.5, ChildSize: 1},
		{Parent: 3, Child: 2, LambdaVal: 0.5, ChildSize: 1},
	}

	selected := map[int]bool{3: true}
	labels, _ := GetLabelsAndProbabilities(tree, selected, 3, false, 0.0, false)

	// All points noise
	for i, l := range labels {
		if l != -1 {
			t.Errorf("labels[%d] = %d, want -1 (all noise when single cluster not allowed)", i, l)
		}
	}
}

func TestGetLabelsAndProbabilities_ProbabilitiesInRange(t *testing.T) {
	tree := []CondensedTreeEntry{
		{Parent: 5, Child: 6, LambdaVal: 0.5, ChildSize: 3},
		{Parent: 5, Child: 7, LambdaVal: 0.5, ChildSize: 2},
		{Parent: 6, Child: 0, LambdaVal: 0.6, ChildSize: 1},
		{Parent: 6, Child: 1, LambdaVal: 0.8, ChildSize: 1},
		{Parent: 6, Child: 2, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 3, LambdaVal: 0.7, ChildSize: 1},
		{Parent: 7, Child: 4, LambdaVal: 1.0, ChildSize: 1},
	}

	selected := map[int]bool{6: true, 7: true}
	labels, probs := GetLabelsAndProbabilities(tree, selected, 5, false, 0.0, false)

	for i := range labels {
		if probs[i] < 0 || probs[i] > 1 {
			t.Errorf("probs[%d] = %f, expected in [0, 1]", i, probs[i])
		}
	}

	// Point 0 in cluster 6: maxLambda=1.0, lambda=0.6 → prob=0.6
	assertFloat(t, "probs[0]", probs[0], 0.6, 1e-10)
	// Point 1 in cluster 6: lambda=0.8 → prob=0.8
	assertFloat(t, "probs[1]", probs[1], 0.8, 1e-10)
	// Point 2 in cluster 6: lambda=1.0 → prob=1.0
	assertFloat(t, "probs[2]", probs[2], 1.0, 1e-10)
	// Point 3 in cluster 7: maxLambda=1.0, lambda=0.7 → prob=0.7
	assertFloat(t, "probs[3]", probs[3], 0.7, 1e-10)
	// Point 4 in cluster 7: lambda=1.0 → prob=1.0
	assertFloat(t, "probs[4]", probs[4], 1.0, 1e-10)
}
