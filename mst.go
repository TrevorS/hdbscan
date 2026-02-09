package hdbscan

import (
	"log"
	"math"
)

// PrimMST computes a minimum spanning tree using Prim's algorithm on a dense
// mutual reachability distance matrix. mrMatrix is flat []float64, n×n row-major.
// Returns (n-1) edges as [][3]float64 where each edge is [from, to, weight].
// Logs a warning if any MST edge weight is +Inf.
func PrimMST(mrMatrix []float64, n int) [][3]float64 {
	if n <= 1 {
		return nil
	}

	inTree := make([]bool, n)
	currentDistances := make([]float64, n)

	// Start from node 0: seed distances from its row in the matrix.
	inTree[0] = true
	currentNode := 0
	currentDistances[0] = math.Inf(1) // node 0 is in tree, distance irrelevant
	for j := 1; j < n; j++ {
		currentDistances[j] = mrMatrix[j]
	}

	edges := make([][3]float64, 0, n-1)
	hasInf := false

	for i := 0; i < n-1; i++ {
		// Find the nearest node not yet in the tree.
		minDist := math.Inf(1)
		minNode := -1
		for j := 0; j < n; j++ {
			if !inTree[j] && currentDistances[j] < minDist {
				minDist = currentDistances[j]
				minNode = j
			}
		}

		// If no finite-distance node was found, pick the first non-tree node.
		// This handles disconnected components (+Inf edges).
		if minNode == -1 {
			for j := 0; j < n; j++ {
				if !inTree[j] {
					minNode = j
					minDist = currentDistances[j]
					break
				}
			}
		}

		if math.IsInf(minDist, 1) {
			hasInf = true
		}

		// Record edge as (currentNode, minNode, weight). currentNode is
		// the previously added node (chain format), matching the reference
		// implementation's mst_linkage_core output.
		edges = append(edges, [3]float64{
			float64(currentNode),
			float64(minNode),
			minDist,
		})

		inTree[minNode] = true
		currentNode = minNode

		// Update distances for remaining non-tree nodes.
		for k := 0; k < n; k++ {
			if !inTree[k] {
				d := mrMatrix[minNode*n+k]
				if d < currentDistances[k] {
					currentDistances[k] = d
				}
			}
		}
	}

	if hasInf {
		log.Printf("hdbscan: MST contains edge(s) with +Inf weight (disconnected components)")
	}

	return edges
}
