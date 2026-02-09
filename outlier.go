package hdbscan

import "math"

// OutlierScores computes GLOSH outlier scores for each point from the condensed tree.
// n is the number of data points.
// Returns scores in [0, 1] for each point.
func OutlierScores(tree []CondensedTreeEntry, n int) []float64 {
	if len(tree) == 0 {
		return make([]float64, n)
	}

	deaths := computeMaxLambdas(tree)
	rootCluster := treeRoot(tree)

	result := make([]float64, n)
	for _, e := range tree {
		point := e.Child
		if point >= rootCluster {
			continue
		}

		lambdaMax := deaths[e.Parent]
		if lambdaMax == 0.0 || math.IsInf(e.LambdaVal, 0) {
			result[point] = 0.0
		} else {
			result[point] = (lambdaMax - e.LambdaVal) / lambdaMax
		}
	}

	return result
}
