package hdbscan

import "math"

// ComputeStability computes stability scores for each cluster in the condensed tree.
//
// The stability of a cluster C is:
//
//	sum over entries with Parent==C of: (entry.LambdaVal - lambdaBirth(C)) * entry.ChildSize
//
// where lambdaBirth(C) is the minimum lambda at which C appears as a child.
// The root cluster (smallest parent ID) has lambdaBirth = 0.
func ComputeStability(tree []CondensedTreeEntry) map[int]float64 {
	if len(tree) == 0 {
		return nil
	}

	// Find the root (smallest parent ID) and compute lambdaBirth for each cluster.
	root := math.MaxInt
	births := make(map[int]float64)
	for _, e := range tree {
		if e.Parent < root {
			root = e.Parent
		}
		if existing, ok := births[e.Child]; !ok || e.LambdaVal < existing {
			births[e.Child] = e.LambdaVal
		}
	}
	births[root] = 0.0

	// Accumulate stability for each parent cluster.
	// Initialize all parents so clusters with only cluster-children still appear.
	stability := make(map[int]float64)
	for _, e := range tree {
		stability[e.Parent] += (e.LambdaVal - births[e.Parent]) * float64(e.ChildSize)
	}

	return stability
}
