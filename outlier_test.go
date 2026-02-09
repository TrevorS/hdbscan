package hdbscan

import (
	"math"
	"testing"
)

func TestOutlierScores_PointAtMaxLambda(t *testing.T) {
	// Point at cluster's maxLambda → score 0
	tree := []CondensedTreeEntry{
		{Parent: 3, Child: 0, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 3, Child: 1, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 3, Child: 2, LambdaVal: 1.0, ChildSize: 1},
	}

	scores := OutlierScores(tree, 3)

	for i, s := range scores {
		if s != 0.0 {
			t.Errorf("scores[%d] = %f, want 0.0 (point at maxLambda)", i, s)
		}
	}
}

func TestOutlierScores_EarlyDeparture(t *testing.T) {
	// Point departing early (low lambda) from a parent cluster.
	// Outlier scores use per-cluster maxLambda (no upward propagation).
	tree := []CondensedTreeEntry{
		{Parent: 4, Child: 5, LambdaVal: 0.2, ChildSize: 3},
		{Parent: 4, Child: 0, LambdaVal: 0.1, ChildSize: 1}, // very early departure
		{Parent: 5, Child: 1, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 5, Child: 2, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 5, Child: 3, LambdaVal: 1.0, ChildSize: 1},
	}

	scores := OutlierScores(tree, 4)

	// maxLambda for cluster 4 = max(0.2, 0.1) = 0.2 (direct children only)
	// maxLambda for cluster 5 = max(1.0, 1.0, 1.0) = 1.0
	//
	// Point 0: cluster=4, maxLambda=0.2, lambda=0.1
	//   score = (0.2 - 0.1) / 0.2 = 0.5
	assertFloat(t, "scores[0]", scores[0], 0.5, 1e-10)

	// Points 1,2,3: cluster=5, maxLambda=1.0, lambda=1.0
	//   score = (1.0 - 1.0) / 1.0 = 0.0
	for i := 1; i <= 3; i++ {
		assertFloat(t, "scores[i]", scores[i], 0.0, 1e-10)
	}
}

func TestOutlierScores_AllInRange(t *testing.T) {
	tree := []CondensedTreeEntry{
		{Parent: 5, Child: 6, LambdaVal: 0.5, ChildSize: 3},
		{Parent: 5, Child: 7, LambdaVal: 0.5, ChildSize: 2},
		{Parent: 6, Child: 0, LambdaVal: 0.6, ChildSize: 1},
		{Parent: 6, Child: 1, LambdaVal: 0.8, ChildSize: 1},
		{Parent: 6, Child: 2, LambdaVal: 1.0, ChildSize: 1},
		{Parent: 7, Child: 3, LambdaVal: 0.7, ChildSize: 1},
		{Parent: 7, Child: 4, LambdaVal: 1.0, ChildSize: 1},
	}

	scores := OutlierScores(tree, 5)

	for i, s := range scores {
		if s < 0 || s > 1 {
			t.Errorf("scores[%d] = %f, expected in [0, 1]", i, s)
		}
	}

	// Cluster 6: maxLambda=1.0
	// Point 0: (1.0-0.6)/1.0 = 0.4
	assertFloat(t, "scores[0]", scores[0], 0.4, 1e-10)
	// Point 1: (1.0-0.8)/1.0 = 0.2
	assertFloat(t, "scores[1]", scores[1], 0.2, 1e-10)
	// Point 2: (1.0-1.0)/1.0 = 0.0
	assertFloat(t, "scores[2]", scores[2], 0.0, 1e-10)

	// Cluster 7: maxLambda=1.0
	// Point 3: (1.0-0.7)/1.0 = 0.3
	assertFloat(t, "scores[3]", scores[3], 0.3, 1e-10)
	// Point 4: (1.0-1.0)/1.0 = 0.0
	assertFloat(t, "scores[4]", scores[4], 0.0, 1e-10)
}

func TestOutlierScores_InfLambda(t *testing.T) {
	// When lambda is +Inf (zero distance), score should be 0
	tree := []CondensedTreeEntry{
		{Parent: 3, Child: 0, LambdaVal: math.Inf(1), ChildSize: 1},
		{Parent: 3, Child: 1, LambdaVal: math.Inf(1), ChildSize: 1},
		{Parent: 3, Child: 2, LambdaVal: 0.5, ChildSize: 1},
	}

	scores := OutlierScores(tree, 3)

	// Points with +Inf lambda → score = 0
	if scores[0] != 0.0 {
		t.Errorf("scores[0] = %f, want 0.0 for +Inf lambda", scores[0])
	}
	if scores[1] != 0.0 {
		t.Errorf("scores[1] = %f, want 0.0 for +Inf lambda", scores[1])
	}
}

func TestOutlierScores_ZeroMaxLambda(t *testing.T) {
	// Edge case: maxLambda = 0 (shouldn't normally happen but handle gracefully)
	tree := []CondensedTreeEntry{
		{Parent: 2, Child: 0, LambdaVal: 0.0, ChildSize: 1},
		{Parent: 2, Child: 1, LambdaVal: 0.0, ChildSize: 1},
	}

	scores := OutlierScores(tree, 2)

	// maxLambda = 0 → score = 0
	for i, s := range scores {
		if s != 0.0 {
			t.Errorf("scores[%d] = %f, want 0.0 for zero maxLambda", i, s)
		}
	}
}
