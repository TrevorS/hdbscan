package hdbscan

// MutualReachability computes the mutual reachability distance matrix.
// distMatrix and coreDistances are flat []float64. n is the number of points.
// alpha scales raw distances: mr[i,j] = max(core[i], core[j], dist[i,j]/alpha).
// When alpha == 1.0, the division is skipped.
// Returns flat []float64 of length n*n.
func MutualReachability(distMatrix, coreDistances []float64, n int, alpha float64) []float64 {
	result := make([]float64, n*n)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			d := distMatrix[i*n+j]
			if alpha != 1.0 {
				d /= alpha
			}
			result[i*n+j] = max(d, coreDistances[i], coreDistances[j])
		}
	}

	return result
}
