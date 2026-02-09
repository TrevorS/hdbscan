package hdbscan

import (
	"math"
	"testing"
)

func TestComputeStability_SmallTree(t *testing.T) {
	// Build a condensed tree from the 6-point example with minClusterSize=2.
	// Clusters: 6 (root), 7, 8, 9, 10
	//
	// Cluster 6 (root): lambdaBirth=0
	//   - Has no direct point children
	//   - stability = 0
	//
	// Cluster 7: lambdaBirth = 0.2 (appears as child of 6 at lambda=0.2)
	//   - Points: 4 (lambda=0.5), 5 (lambda=0.5)
	//   - stability = (0.5 - 0.2) + (0.5 - 0.2) = 0.3 + 0.3 = 0.6
	//
	// Cluster 8: lambdaBirth = 0.2 (appears as child of 6 at lambda=0.2)
	//   - Has no direct point children (only cluster children 9, 10)
	//   - stability = 0
	//
	// Cluster 9: lambdaBirth = 1/3 (appears as child of 8 at lambda=1/3)
	//   - Points: 0 (lambda=1.0), 1 (lambda=1.0)
	//   - stability = (1.0 - 1/3) + (1.0 - 1/3) = 2/3 + 2/3 = 4/3
	//
	// Cluster 10: lambdaBirth = 1/3 (appears as child of 8 at lambda=1/3)
	//   - Points: 2 (lambda=2/3), 3 (lambda=2/3)
	//   - stability = (2/3 - 1/3) + (2/3 - 1/3) = 1/3 + 1/3 = 2/3
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

	const eps = 1e-10

	// Root cluster 6: lambdaBirth=0
	// Entries: (6,7,0.2,2) → (0.2-0)*2=0.4; (6,8,0.2,4) → (0.2-0)*4=0.8
	// stability = 1.2
	assertFloat(t, "stability[6]", stab[6], 1.2, eps)

	// Cluster 7: lambdaBirth=0.2, points at 0.5
	assertFloat(t, "stability[7]", stab[7], 0.6, eps)

	// Cluster 8: lambdaBirth=0.2
	// Entries: (8,9,1/3,2) → (1/3-0.2)*2=4/15; (8,10,1/3,2) → (1/3-0.2)*2=4/15
	// stability = 8/15 ≈ 0.5333
	assertFloat(t, "stability[8]", stab[8], 8.0/15.0, eps)

	// Cluster 9: lambdaBirth=1/3, points at 1.0
	assertFloat(t, "stability[9]", stab[9], 4.0/3.0, eps)

	// Cluster 10: lambdaBirth=1/3, points at 2/3
	assertFloat(t, "stability[10]", stab[10], 2.0/3.0, eps)

	// All clusters should have entries
	if len(stab) != 5 {
		t.Errorf("expected 5 cluster stabilities, got %d", len(stab))
	}
}

func TestComputeStability_RootLambdaBirthZero(t *testing.T) {
	// Minimal tree: root cluster with just points
	tree := []CondensedTreeEntry{
		{Parent: 3, Child: 0, LambdaVal: 0.5, ChildSize: 1},
		{Parent: 3, Child: 1, LambdaVal: 0.5, ChildSize: 1},
		{Parent: 3, Child: 2, LambdaVal: 0.5, ChildSize: 1},
	}

	stab := ComputeStability(tree)

	// Root cluster 3: lambdaBirth=0, points at 0.5
	// stability = 3 * (0.5 - 0) = 1.5
	assertFloat(t, "stability[3]", stab[3], 1.5, 1e-10)
}

func TestComputeStability_UniformDensity(t *testing.T) {
	// Cluster with all points at the same lambda
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

	const eps = 1e-10

	// Root cluster 5: lambdaBirth=0
	// Entries: (5,6,0.5,3) → 0.5*3=1.5; (5,7,0.5,2) → 0.5*2=1.0
	// stability = 2.5
	assertFloat(t, "stability[5]", stab[5], 2.5, eps)

	// Cluster 6: lambdaBirth=0.5, 3 points at lambda=1.0
	// stability = 3 * (1.0 - 0.5) = 1.5
	assertFloat(t, "stability[6]", stab[6], 1.5, eps)

	// Cluster 7: lambdaBirth=0.5, 2 points at lambda=1.0
	// stability = 2 * (1.0 - 0.5) = 1.0
	assertFloat(t, "stability[7]", stab[7], 1.0, eps)
}

func assertFloat(t *testing.T, name string, got, want, eps float64) {
	t.Helper()
	if math.Abs(got-want) > eps {
		t.Errorf("%s: got %f, want %f", name, got, want)
	}
}
