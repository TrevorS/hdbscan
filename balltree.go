package hdbscan

import (
	"container/heap"
	"math"
	"sort"
)

// BallTree is a ball tree spatial index for nearest-neighbor queries and
// Borůvka MST acceleration. Each node stores a centroid and radius defining
// the smallest enclosing ball for its points.
//
// The tree is stored as a complete binary tree in array form:
//   - node i has children at 2*i+1 and 2*i+2
//   - precomputed centroid distances enable O(1) MinRdistDual
type BallTree struct {
	data     []float64 // flat row-major point data (n * dims)
	n        int       // number of points
	dims     int       // dimensionality
	leafSize int
	metric   DistanceMetric
	idxArray []int      // permutation: tree-order position → original index
	nodes    []NodeData // one entry per tree node; Radius is used
	// centroids[node*dims .. (node+1)*dims) = centroid of node
	centroids []float64
	// centroidDists[i*numNodesAlloc + j] = metric.Distance(centroid_i, centroid_j)
	centroidDists []float64
	numNodes      int
	numNodesAlloc int // allocated width of centroidDists
}

// NewBallTree builds a ball tree from flat row-major data with n points
// of dimensionality dims. leafSize controls the max points per leaf node.
func NewBallTree(data []float64, n, dims int, metric DistanceMetric, leafSize int) *BallTree {
	if leafSize < 1 {
		leafSize = 1
	}

	dataCopy := make([]float64, len(data))
	copy(dataCopy, data)
	idxArray := make([]int, n)
	for i := range idxArray {
		idxArray[i] = i
	}

	maxNodes := kdMaxNodes(n, leafSize) // reuse the same upper bound
	t := &BallTree{
		data:          dataCopy,
		n:             n,
		dims:          dims,
		leafSize:      leafSize,
		metric:        metric,
		idxArray:      idxArray,
		nodes:         make([]NodeData, maxNodes),
		centroids:     make([]float64, maxNodes*dims),
		numNodesAlloc: maxNodes,
	}

	if n > 0 {
		t.buildNode(0, 0, n)
		t.numNodes = btCountNodes(t.nodes, 0, maxNodes)
		t.precomputeCentroidDists()
	}

	return t
}

// btCountNodes counts how many nodes were actually initialized.
func btCountNodes(nodes []NodeData, nodeID, maxNodes int) int {
	if nodeID >= maxNodes {
		return 0
	}
	if nodes[nodeID].IdxStart == 0 && nodes[nodeID].IdxEnd == 0 && nodeID != 0 {
		return 0
	}
	count := 1
	if !nodes[nodeID].IsLeaf {
		count += btCountNodes(nodes, 2*nodeID+1, maxNodes)
		count += btCountNodes(nodes, 2*nodeID+2, maxNodes)
	}
	return count
}

// buildNode recursively builds the ball tree for points in idxArray[start:end].
func (t *BallTree) buildNode(nodeID, start, end int) {
	for nodeID >= len(t.nodes) {
		t.nodes = append(t.nodes, NodeData{})
		t.centroids = append(t.centroids, make([]float64, t.dims)...)
		t.numNodesAlloc = len(t.nodes)
	}

	// Compute centroid.
	t.computeCentroid(nodeID, start, end)

	// Compute radius: max distance from centroid to any point in this node.
	centroid := t.centroids[nodeID*t.dims : (nodeID+1)*t.dims]
	var radius float64
	for i := start; i < end; i++ {
		ptIdx := t.idxArray[i]
		pt := t.data[ptIdx*t.dims : (ptIdx+1)*t.dims]
		d := t.metric.Distance(centroid, pt)
		if d > radius {
			radius = d
		}
	}

	count := end - start
	if count <= t.leafSize {
		t.nodes[nodeID] = NodeData{IdxStart: start, IdxEnd: end, IsLeaf: true, Radius: radius}
		return
	}

	// Partition: find the point farthest from centroid, then the point
	// farthest from that. Split by projection onto the axis between them.
	t.nodes[nodeID] = NodeData{IdxStart: start, IdxEnd: end, IsLeaf: false, Radius: radius}

	// Find dimension with greatest spread (simple partitioning strategy
	// that works well in practice and matches sklearn's approach for
	// moderate dimensionality).
	splitDim := t.findSpreadDim(start, end)
	t.sortByDim(start, end, splitDim)
	mid := start + count/2

	t.buildNode(2*nodeID+1, start, mid)
	t.buildNode(2*nodeID+2, mid, end)
}

// computeCentroid computes the mean of points idxArray[start:end] and stores
// it in the centroids array.
func (t *BallTree) computeCentroid(nodeID, start, end int) {
	base := nodeID * t.dims
	count := float64(end - start)
	for d := 0; d < t.dims; d++ {
		t.centroids[base+d] = 0
	}
	for i := start; i < end; i++ {
		ptIdx := t.idxArray[i]
		for d := 0; d < t.dims; d++ {
			t.centroids[base+d] += t.data[ptIdx*t.dims+d]
		}
	}
	for d := 0; d < t.dims; d++ {
		t.centroids[base+d] /= count
	}
}

// findSpreadDim returns the dimension with the greatest spread among
// points in idxArray[start:end].
func (t *BallTree) findSpreadDim(start, end int) int {
	bestDim := 0
	bestSpread := -1.0
	for d := 0; d < t.dims; d++ {
		minVal := math.Inf(1)
		maxVal := math.Inf(-1)
		for i := start; i < end; i++ {
			v := t.data[t.idxArray[i]*t.dims+d]
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
		spread := maxVal - minVal
		if spread > bestSpread {
			bestSpread = spread
			bestDim = d
		}
	}
	return bestDim
}

// sortByDim sorts idxArray[start:end] by the given dimension.
func (t *BallTree) sortByDim(start, end, dim int) {
	sub := t.idxArray[start:end]
	dims := t.dims
	data := t.data
	sort.Slice(sub, func(i, j int) bool {
		return data[sub[i]*dims+dim] < data[sub[j]*dims+dim]
	})
}

// precomputeCentroidDists builds the pairwise centroid distance matrix.
func (t *BallTree) precomputeCentroidDists() {
	nn := t.numNodesAlloc
	t.centroidDists = make([]float64, nn*nn)
	for i := 0; i < t.numNodes; i++ {
		ci := t.centroids[i*t.dims : (i+1)*t.dims]
		for j := i + 1; j < t.numNodes; j++ {
			cj := t.centroids[j*t.dims : (j+1)*t.dims]
			d := t.metric.Distance(ci, cj)
			t.centroidDists[i*nn+j] = d
			t.centroidDists[j*nn+i] = d
		}
	}
}

// --- SpatialTree interface ---

func (t *BallTree) Data() []float64           { return t.data }
func (t *BallTree) NumPoints() int            { return t.n }
func (t *BallTree) NumFeatures() int          { return t.dims }
func (t *BallTree) IdxArray() []int           { return t.idxArray }
func (t *BallTree) NodeDataArray() []NodeData { return t.nodes[:t.numNodes] }

// QueryKNN finds the k nearest neighbors for each row in queryData.
func (t *BallTree) QueryKNN(queryData []float64, queryRows, k int) ([][]int, [][]float64) {
	indices := make([][]int, queryRows)
	distances := make([][]float64, queryRows)

	for q := 0; q < queryRows; q++ {
		query := queryData[q*t.dims : (q+1)*t.dims]
		h := &knnHeap{}
		heap.Init(h)
		t.knnSearch(0, query, k, h)

		nResults := h.Len()
		idx := make([]int, nResults)
		dist := make([]float64, nResults)
		for i := nResults - 1; i >= 0; i-- {
			item := heap.Pop(h).(knnItem)
			idx[i] = item.index
			dist[i] = item.dist
		}
		indices[q] = idx
		distances[q] = dist
	}

	return indices, distances
}

// knnSearch performs a single-tree KNN traversal for the ball tree.
func (t *BallTree) knnSearch(nodeID int, query []float64, k int, h *knnHeap) {
	if nodeID >= len(t.nodes) {
		return
	}
	node := t.nodes[nodeID]
	if node.IdxStart == node.IdxEnd && nodeID != 0 {
		return
	}

	if node.IsLeaf {
		for i := node.IdxStart; i < node.IdxEnd; i++ {
			ptIdx := t.idxArray[i]
			pt := t.data[ptIdx*t.dims : (ptIdx+1)*t.dims]
			d := t.metric.Distance(query, pt)
			if h.Len() < k {
				heap.Push(h, knnItem{index: ptIdx, dist: d})
			} else if d < (*h)[0].dist {
				(*h)[0] = knnItem{index: ptIdx, dist: d}
				heap.Fix(h, 0)
			}
		}
		return
	}

	left := 2*nodeID + 1
	right := 2*nodeID + 2

	// Use centroid distance to query minus radius as lower bound.
	centroidL := t.centroids[left*t.dims : (left+1)*t.dims]
	centroidR := t.centroids[right*t.dims : (right+1)*t.dims]
	leftDist := t.metric.Distance(query, centroidL) - t.nodes[left].Radius
	rightDist := t.metric.Distance(query, centroidR) - t.nodes[right].Radius
	if leftDist < 0 {
		leftDist = 0
	}
	if rightDist < 0 {
		rightDist = 0
	}

	nearChild, farChild := left, right
	farDist := rightDist
	if rightDist < leftDist {
		nearChild, farChild = right, left
		farDist = leftDist
	}

	t.knnSearch(nearChild, query, k, h)

	if h.Len() < k || farDist < (*h)[0].dist {
		t.knnSearch(farChild, query, k, h)
	}
}

// --- BoruvkaTree interface ---

func (t *BallTree) NumNodes() int { return t.numNodes }

func (t *BallTree) ChildNodes(node int) (left, right int) {
	return 2*node + 1, 2*node + 2
}

// MinRdistDual returns a lower bound in reduced-distance space on the
// distance between any point in node1 and any point in node2.
// Uses precomputed centroid distances: max(0, centroidDist - r1 - r2),
// converted to reduced distance.
func (t *BallTree) MinRdistDual(node1, node2 int) float64 {
	nn := t.numNodesAlloc
	dist := t.centroidDists[node1*nn+node2] - t.nodes[node1].Radius - t.nodes[node2].Radius
	if dist < 0 {
		dist = 0
	}
	return t.metric.DistToRdist(dist)
}

// MinRdistPoint returns a lower bound in reduced-distance space on the
// distance between a point and any point in the given node.
func (t *BallTree) MinRdistPoint(node int, point []float64) float64 {
	centroid := t.centroids[node*t.dims : (node+1)*t.dims]
	dist := t.metric.Distance(point, centroid) - t.nodes[node].Radius
	if dist < 0 {
		dist = 0
	}
	return t.metric.DistToRdist(dist)
}
