package hdbscan

import "sort"

// ComputeCoreDistances computes core distances from a distance matrix.
// distMatrix is flat n*n row-major. minSamples is clamped to [0, n-1].
// Returns []float64 of length n where core[i] is the distance to the
// minSamples-th nearest neighbor of point i.
func ComputeCoreDistances(distMatrix []float64, n, minSamples int) []float64 {
	minSamples = min(minSamples, n-1)
	minSamples = max(minSamples, 0)

	core := make([]float64, n)
	if minSamples == 0 {
		return core
	}

	for i := 0; i < n; i++ {
		neighbors := make([]float64, 0, n-1)
		for j := 0; j < n; j++ {
			if j != i {
				neighbors = append(neighbors, distMatrix[i*n+j])
			}
		}
		sort.Float64s(neighbors)
		core[i] = neighbors[minSamples-1]
	}

	return core
}
