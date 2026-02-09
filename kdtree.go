package hdbscan

import (
	"container/heap"
	"math"
	"sort"
)

// KDTree is a KD-tree spatial index for nearest-neighbor queries and
// Borůvka MST acceleration. Points are stored in a flat row-major array
// and reordered internally via an index permutation array.
//
// The tree is stored as a complete binary tree in array form:
//   - node i has children at 2*i+1 and 2*i+2
//   - node bounds are stored as min/max per dimension per node
type KDTree struct {
	data     []float64 // flat row-major point data (n * dims)
	n        int       // number of points
	dims     int       // dimensionality
	leafSize int
	metric   DistanceMetric
	idxArray []int      // permutation: tree-order position → original index
	nodes    []NodeData // one entry per tree node
	// nodeBoundsMin[node*dims + j] = min value of feature j in node
	nodeBoundsMin []float64
	// nodeBoundsMax[node*dims + j] = max value of feature j in node
	nodeBoundsMax []float64
	numNodes      int
}

// NewKDTree builds a KD-tree from flat row-major data with n points of
// dimensionality dims. leafSize controls the max points per leaf node.
func NewKDTree(data []float64, n, dims int, metric DistanceMetric, leafSize int) *KDTree {
	if leafSize < 1 {
		leafSize = 1
	}

	// Copy data and build identity index array.
	dataCopy := make([]float64, len(data))
	copy(dataCopy, data)
	idxArray := make([]int, n)
	for i := range idxArray {
		idxArray[i] = i
	}

	// Pre-allocate tree arrays. A complete binary tree with n leaves of
	// size leafSize needs at most 2*ceil(n/leafSize) nodes, but we use
	// a generous upper bound since the median split may not be perfectly balanced.
	maxNodes := kdMaxNodes(n, leafSize)

	t := &KDTree{
		data:          dataCopy,
		n:             n,
		dims:          dims,
		leafSize:      leafSize,
		metric:        metric,
		idxArray:      idxArray,
		nodes:         make([]NodeData, maxNodes),
		nodeBoundsMin: make([]float64, maxNodes*dims),
		nodeBoundsMax: make([]float64, maxNodes*dims),
	}

	if n > 0 {
		t.buildNode(0, 0, n)
		t.numNodes = kdCountNodes(t.nodes, 0, maxNodes)
	}

	return t
}

// kdMaxNodes returns an upper bound on the number of nodes needed for a
// binary tree with n points and the given leaf size.
func kdMaxNodes(n, leafSize int) int {
	if n == 0 {
		return 1
	}
	// Depth of tree: ceil(log2(ceil(n/leafSize))) + 1.
	// Number of nodes in a complete binary tree of depth d = 2^(d+1) - 1.
	leaves := (n + leafSize - 1) / leafSize
	depth := 0
	v := 1
	for v < leaves {
		v *= 2
		depth++
	}
	return (1 << (depth + 1)) - 1 + 2 // +2 for safety margin
}

// kdCountNodes counts how many nodes were actually initialized by the build.
func kdCountNodes(nodes []NodeData, nodeID, maxNodes int) int {
	if nodeID >= maxNodes {
		return 0
	}
	if nodes[nodeID].IdxStart == 0 && nodes[nodeID].IdxEnd == 0 && nodeID != 0 {
		return 0
	}
	count := 1
	left := 2*nodeID + 1
	right := 2*nodeID + 2
	if !nodes[nodeID].IsLeaf {
		count += kdCountNodes(nodes, left, maxNodes)
		count += kdCountNodes(nodes, right, maxNodes)
	}
	return count
}

// buildNode recursively builds the tree for points in idxArray[start:end].
func (t *KDTree) buildNode(nodeID, start, end int) {
	// Grow arrays if needed (shouldn't happen with good upper bound).
	for nodeID >= len(t.nodes) {
		t.nodes = append(t.nodes, NodeData{})
		t.nodeBoundsMin = append(t.nodeBoundsMin, make([]float64, t.dims)...)
		t.nodeBoundsMax = append(t.nodeBoundsMax, make([]float64, t.dims)...)
	}

	// Compute bounds for this node.
	t.computeNodeBounds(nodeID, start, end)

	count := end - start
	if count <= t.leafSize {
		t.nodes[nodeID] = NodeData{IdxStart: start, IdxEnd: end, IsLeaf: true}
		return
	}

	// Find dimension with greatest spread.
	splitDim := 0
	maxSpread := -1.0
	for d := 0; d < t.dims; d++ {
		spread := t.nodeBoundsMax[nodeID*t.dims+d] - t.nodeBoundsMin[nodeID*t.dims+d]
		if spread > maxSpread {
			maxSpread = spread
			splitDim = d
		}
	}

	// Sort by the split dimension and split at the median.
	t.sortByDimension(start, end, splitDim)
	mid := start + count/2

	t.nodes[nodeID] = NodeData{IdxStart: start, IdxEnd: end, IsLeaf: false}

	t.buildNode(2*nodeID+1, start, mid)
	t.buildNode(2*nodeID+2, mid, end)
}

// computeNodeBounds computes min/max per dimension for points idxArray[start:end].
func (t *KDTree) computeNodeBounds(nodeID, start, end int) {
	base := nodeID * t.dims
	for d := 0; d < t.dims; d++ {
		t.nodeBoundsMin[base+d] = math.Inf(1)
		t.nodeBoundsMax[base+d] = math.Inf(-1)
	}
	for i := start; i < end; i++ {
		ptIdx := t.idxArray[i]
		for d := 0; d < t.dims; d++ {
			v := t.data[ptIdx*t.dims+d]
			if v < t.nodeBoundsMin[base+d] {
				t.nodeBoundsMin[base+d] = v
			}
			if v > t.nodeBoundsMax[base+d] {
				t.nodeBoundsMax[base+d] = v
			}
		}
	}
}

// sortByDimension sorts idxArray[start:end] by the given dimension.
func (t *KDTree) sortByDimension(start, end, dim int) {
	sub := t.idxArray[start:end]
	dims := t.dims
	data := t.data
	sort.Slice(sub, func(i, j int) bool {
		return data[sub[i]*dims+dim] < data[sub[j]*dims+dim]
	})
}

// --- SpatialTree interface ---

func (t *KDTree) Data() []float64           { return t.data }
func (t *KDTree) NumPoints() int            { return t.n }
func (t *KDTree) NumFeatures() int          { return t.dims }
func (t *KDTree) IdxArray() []int           { return t.idxArray }
func (t *KDTree) NodeDataArray() []NodeData { return t.nodes[:t.numNodes] }

// QueryKNN finds the k nearest neighbors for each row in queryData.
func (t *KDTree) QueryKNN(queryData []float64, queryRows, k int) ([][]int, [][]float64) {
	indices := make([][]int, queryRows)
	distances := make([][]float64, queryRows)

	for q := 0; q < queryRows; q++ {
		query := queryData[q*t.dims : (q+1)*t.dims]
		h := &knnHeap{}
		heap.Init(h)
		t.knnSearch(0, query, k, h)

		// Extract results sorted by distance (ascending).
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

// knnSearch performs a single-tree KNN traversal using a max-heap of size k.
func (t *KDTree) knnSearch(nodeID int, query []float64, k int, h *knnHeap) {
	if nodeID >= len(t.nodes) {
		return
	}
	node := t.nodes[nodeID]
	if node.IdxStart == node.IdxEnd && nodeID != 0 {
		return // uninitialized node
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

	// Determine which child to visit first (nearer child first).
	left := 2*nodeID + 1
	right := 2*nodeID + 2

	leftRdist := t.minRdistPointInternal(left, query)
	rightRdist := t.minRdistPointInternal(right, query)

	nearChild, farChild := left, right
	farRdist := rightRdist
	if rightRdist < leftRdist {
		nearChild, farChild = right, left
		farRdist = leftRdist
	}

	t.knnSearch(nearChild, query, k, h)

	// Prune far child if its lower bound exceeds the current k-th distance.
	if h.Len() < k || t.metric.DistToRdist((*h)[0].dist) > farRdist {
		t.knnSearch(farChild, query, k, h)
	}
}

// --- BoruvkaTree interface ---

func (t *KDTree) NumNodes() int { return t.numNodes }

func (t *KDTree) ChildNodes(node int) (left, right int) {
	return 2*node + 1, 2*node + 2
}

// MinRdistDual returns a lower bound in reduced-distance space on the
// distance between any point in node1 and any point in node2.
// For axis-aligned boxes, this computes the per-dimension gap and
// aggregates according to the metric.
func (t *KDTree) MinRdistDual(node1, node2 int) float64 {
	return t.minRdistDualInternal(node1, node2)
}

func (t *KDTree) minRdistDualInternal(node1, node2 int) float64 {
	dims := t.dims
	base1 := node1 * dims
	base2 := node2 * dims

	switch m := t.metric.(type) {
	case ChebyshevMetric:
		var rdist float64
		for j := 0; j < dims; j++ {
			// Gap between boxes along dimension j.
			d1 := t.nodeBoundsMin[base1+j] - t.nodeBoundsMax[base2+j]
			d2 := t.nodeBoundsMin[base2+j] - t.nodeBoundsMax[base1+j]
			// max(d1, d2, 0)
			d := math.Max(d1, math.Max(d2, 0))
			if d > rdist {
				rdist = d
			}
		}
		return rdist

	case MinkowskiMetric:
		var rdist float64
		for j := 0; j < dims; j++ {
			d1 := t.nodeBoundsMin[base1+j] - t.nodeBoundsMax[base2+j]
			d2 := t.nodeBoundsMin[base2+j] - t.nodeBoundsMax[base1+j]
			d := math.Max(d1, math.Max(d2, 0))
			rdist += math.Pow(d, m.P)
		}
		return rdist

	default:
		// Euclidean, Manhattan, and others that decompose along axes.
		// For Euclidean: sum of squared per-dim gaps (reduced distance).
		// For Manhattan: sum of per-dim gaps (same as distance).
		var rdist float64
		p := metricP(t.metric)
		for j := 0; j < dims; j++ {
			d1 := t.nodeBoundsMin[base1+j] - t.nodeBoundsMax[base2+j]
			d2 := t.nodeBoundsMin[base2+j] - t.nodeBoundsMax[base1+j]
			d := math.Max(d1, math.Max(d2, 0))
			rdist += math.Pow(d, p)
		}
		return rdist
	}
}

// MinRdistPoint returns a lower bound in reduced-distance space on the
// distance between a point and any point in the given node.
func (t *KDTree) MinRdistPoint(node int, point []float64) float64 {
	return t.minRdistPointInternal(node, point)
}

func (t *KDTree) minRdistPointInternal(node int, point []float64) float64 {
	if node >= len(t.nodes) {
		return math.Inf(1)
	}
	dims := t.dims
	base := node * dims

	switch m := t.metric.(type) {
	case ChebyshevMetric:
		var rdist float64
		for j := 0; j < dims; j++ {
			lo := t.nodeBoundsMin[base+j]
			hi := t.nodeBoundsMax[base+j]
			var d float64
			if point[j] < lo {
				d = lo - point[j]
			} else if point[j] > hi {
				d = point[j] - hi
			}
			if d > rdist {
				rdist = d
			}
		}
		return rdist

	case MinkowskiMetric:
		var rdist float64
		for j := 0; j < dims; j++ {
			lo := t.nodeBoundsMin[base+j]
			hi := t.nodeBoundsMax[base+j]
			var d float64
			if point[j] < lo {
				d = lo - point[j]
			} else if point[j] > hi {
				d = point[j] - hi
			}
			rdist += math.Pow(d, m.P)
		}
		return rdist

	default:
		var rdist float64
		p := metricP(t.metric)
		for j := 0; j < dims; j++ {
			lo := t.nodeBoundsMin[base+j]
			hi := t.nodeBoundsMax[base+j]
			var d float64
			if point[j] < lo {
				d = lo - point[j]
			} else if point[j] > hi {
				d = point[j] - hi
			}
			rdist += math.Pow(d, p)
		}
		return rdist
	}
}

// metricP returns the Minkowski exponent for the metric, defaulting to
// 2 for Euclidean and 1 for Manhattan.
func metricP(m DistanceMetric) float64 {
	switch v := m.(type) {
	case EuclideanMetric:
		return 2.0
	case ManhattanMetric:
		return 1.0
	case MinkowskiMetric:
		return v.P
	case ChebyshevMetric:
		return math.Inf(1)
	default:
		return 2.0 // fallback; Euclidean-like
	}
}

// --- max-heap for KNN queries ---

type knnItem struct {
	index int
	dist  float64
}

// knnHeap is a max-heap of knnItem (largest distance on top) used as a
// bounded priority queue for KNN queries.
type knnHeap []knnItem

func (h knnHeap) Len() int            { return len(h) }
func (h knnHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist } // max-heap
func (h knnHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *knnHeap) Push(x interface{}) { *h = append(*h, x.(knnItem)) }
func (h *knnHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}
