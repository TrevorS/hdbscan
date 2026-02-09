package hdbscan

// ComputeCoreDistancesTree computes core distances using a spatial tree's
// KNN queries instead of a full distance matrix. For each point, it finds
// (minSamples+1) nearest neighbors (including self) and takes the k-th
// neighbor distance as the core distance.
//
// The returned slice has length tree.NumPoints(), where core[i] is the
// true (non-reduced) distance to the minSamples-th nearest non-self neighbor
// of point i.
func ComputeCoreDistancesTree(tree SpatialTree, minSamples int) []float64 {
	n := tree.NumPoints()
	if n == 0 {
		return nil
	}

	minSamples = min(minSamples, n-1)
	minSamples = max(minSamples, 0)

	core := make([]float64, n)
	if minSamples == 0 {
		return core
	}

	// Query k = minSamples+1 neighbors (the +1 accounts for the point itself).
	k := minSamples + 1
	if k > n {
		k = n
	}

	indices, distances := tree.QueryKNN(tree.Data(), n, k)

	for i := 0; i < n; i++ {
		// The KNN result includes the point itself (distance 0). Find the
		// minSamples-th non-self neighbor.
		neighborCount := 0
		for j := 0; j < len(distances[i]); j++ {
			if indices[i][j] == i {
				continue // skip self
			}
			neighborCount++
			if neighborCount == minSamples {
				core[i] = distances[i][j]
				break
			}
		}
		// If we couldn't find enough non-self neighbors (shouldn't happen
		// with correct k), use the last available distance.
		if neighborCount < minSamples && len(distances[i]) > 0 {
			core[i] = distances[i][len(distances[i])-1]
		}
	}

	return core
}
