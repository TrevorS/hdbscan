package hdbscan

import (
	"log"
	"math"
)

// PrimMSTVector computes a minimum spanning tree using Prim's algorithm without
// requiring a precomputed n×n distance matrix. It computes mutual reachability
// distances on-the-fly, using O(n) memory instead of O(n²).
//
// data is flat row-major with n rows and dims columns.
// coreDistances has length n (one core distance per point).
// metric is the distance function. alpha scales raw distances (mr = max(core[i], core[j], dist/alpha)).
//
// Returns (n-1) edges as [][3]float64 where each edge is [from, to, weight].
// The "from" field is the actual nearest tree neighbor (currentSources[newNode]),
// not just the last-added node.
//
// Port of mst_linkage_core_vector from the Python reference implementation.
func PrimMSTVector(data []float64, n, dims int, coreDistances []float64, metric DistanceMetric, alpha float64) [][3]float64 {
	if n <= 1 {
		return nil
	}

	inTree := make([]bool, n)
	currentDistances := make([]float64, n)
	currentSources := make([]int, n)

	// Initialize: all distances to +Inf, sources to 0.
	for j := 0; j < n; j++ {
		currentDistances[j] = math.Inf(1)
	}

	currentNode := 0
	edges := make([][3]float64, 0, n-1)
	hasInf := false

	for i := 1; i < n; i++ {
		inTree[currentNode] = true
		currentNodeCoreDist := coreDistances[currentNode]

		newDistance := math.MaxFloat64
		sourceNode := 0
		newNode := 0

		for j := 0; j < n; j++ {
			if inTree[j] {
				continue
			}

			rightValue := currentDistances[j]
			rightSource := currentSources[j]

			// Compute raw distance between currentNode and j.
			leftValue := metric.Distance(
				data[currentNode*dims:(currentNode+1)*dims],
				data[j*dims:(j+1)*dims],
			)
			leftSource := currentNode

			if alpha != 1.0 {
				leftValue /= alpha
			}

			coreValue := coreDistances[j]

			// Check if the existing best (rightValue) is already smaller than
			// both core distances and the new raw distance. If so, no update needed.
			if currentNodeCoreDist > rightValue ||
				coreValue > rightValue ||
				leftValue > rightValue {
				// rightValue is already the best; just check if it's the global min.
				if rightValue < newDistance {
					newDistance = rightValue
					sourceNode = rightSource
					newNode = j
				}
				continue
			}

			// Apply mutual reachability: leftValue = max(leftValue, core[currentNode], core[j]).
			if coreValue > currentNodeCoreDist {
				if coreValue > leftValue {
					leftValue = coreValue
				}
			} else {
				if currentNodeCoreDist > leftValue {
					leftValue = currentNodeCoreDist
				}
			}

			if leftValue < rightValue {
				// New path through currentNode is better.
				currentDistances[j] = leftValue
				currentSources[j] = leftSource
				if leftValue < newDistance {
					newDistance = leftValue
					sourceNode = leftSource
					newNode = j
				}
			} else {
				// Existing path is still best.
				if rightValue < newDistance {
					newDistance = rightValue
					sourceNode = rightSource
					newNode = j
				}
			}
		}

		if math.IsInf(newDistance, 1) || newDistance == math.MaxFloat64 {
			hasInf = true
		}

		edges = append(edges, [3]float64{
			float64(sourceNode),
			float64(newNode),
			newDistance,
		})

		currentNode = newNode
	}

	if hasInf {
		log.Printf("hdbscan: MST contains edge(s) with +Inf weight (disconnected components)")
	}

	return edges
}
