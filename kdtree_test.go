package hdbscan

import (
	"math"
	"sort"
	"testing"
)

// --- Construction tests ---

func TestKDTree_Construction_BasicProperties(t *testing.T) {
	// 6 points in 2D
	data := []float64{
		0, 0,
		1, 0,
		2, 0,
		0, 3,
		1, 3,
		2, 3,
	}
	n, dims := 6, 2
	tree := NewKDTree(data, n, dims, EuclideanMetric{}, 2)

	if tree.NumPoints() != n {
		t.Errorf("NumPoints() = %d, want %d", tree.NumPoints(), n)
	}
	if tree.NumFeatures() != dims {
		t.Errorf("NumFeatures() = %d, want %d", tree.NumFeatures(), dims)
	}
	if tree.NumNodes() < 1 {
		t.Errorf("NumNodes() = %d, want >= 1", tree.NumNodes())
	}

	// IdxArray should be a permutation of 0..n-1.
	idx := tree.IdxArray()
	if len(idx) != n {
		t.Fatalf("IdxArray length = %d, want %d", len(idx), n)
	}
	seen := make(map[int]bool)
	for _, v := range idx {
		if v < 0 || v >= n {
			t.Errorf("IdxArray contains out-of-range index %d", v)
		}
		if seen[v] {
			t.Errorf("IdxArray contains duplicate index %d", v)
		}
		seen[v] = true
	}
}

func TestKDTree_Construction_LeafSize1(t *testing.T) {
	data := []float64{0, 0, 1, 1, 2, 2, 3, 3}
	tree := NewKDTree(data, 4, 2, EuclideanMetric{}, 1)

	// With leafSize=1, every leaf has exactly 1 point.
	for _, nd := range tree.NodeDataArray() {
		if nd.IsLeaf && (nd.IdxEnd-nd.IdxStart) != 1 {
			t.Errorf("leaf has %d points, want 1", nd.IdxEnd-nd.IdxStart)
		}
	}
}

func TestKDTree_Construction_LeafSizeLargerThanN(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	tree := NewKDTree(data, 2, 2, EuclideanMetric{}, 100)

	// All points fit in one leaf.
	nodes := tree.NodeDataArray()
	if len(nodes) != 1 {
		t.Errorf("expected 1 node for leafSize > n, got %d", len(nodes))
	}
	if !nodes[0].IsLeaf {
		t.Error("root should be a leaf when leafSize > n")
	}
}

func TestKDTree_Construction_SinglePoint(t *testing.T) {
	data := []float64{5, 5}
	tree := NewKDTree(data, 1, 2, EuclideanMetric{}, 10)

	if tree.NumPoints() != 1 {
		t.Errorf("NumPoints() = %d, want 1", tree.NumPoints())
	}
	if tree.NumNodes() != 1 {
		t.Errorf("NumNodes() = %d, want 1", tree.NumNodes())
	}
}

func TestKDTree_Construction_TwoPoints(t *testing.T) {
	data := []float64{0, 0, 10, 10}
	tree := NewKDTree(data, 2, 2, EuclideanMetric{}, 1)

	if tree.NumPoints() != 2 {
		t.Errorf("NumPoints() = %d, want 2", tree.NumPoints())
	}
}

// --- KNN query tests ---

func TestKDTree_KNN_BruteForceMatch(t *testing.T) {
	// 5 points in 2D: compare tree KNN to brute-force.
	data := []float64{
		0, 0,
		3, 0,
		0, 4,
		3, 4,
		1.5, 2,
	}
	n, dims := 5, 2

	for _, metric := range []DistanceMetric{
		EuclideanMetric{},
		ManhattanMetric{},
	} {
		tree := NewKDTree(data, n, dims, metric, 1)
		for k := 1; k <= n; k++ {
			indices, distances := tree.QueryKNN(data, n, k)
			for q := 0; q < n; q++ {
				bruteIdx, bruteDist := bruteForceKNN(data, n, dims, q, k, metric)
				if !knnResultsMatch(indices[q], distances[q], bruteIdx, bruteDist, floatTol) {
					t.Errorf("metric=%T k=%d query=%d: tree KNN doesn't match brute force.\n  tree: idx=%v dist=%v\n  brute: idx=%v dist=%v",
						metric, k, q, indices[q], distances[q], bruteIdx, bruteDist)
				}
			}
		}
	}
}

func TestKDTree_KNN_Minkowski(t *testing.T) {
	data := []float64{
		0, 0,
		1, 0,
		0, 1,
		1, 1,
	}
	n, dims := 4, 2
	metric := MinkowskiMetric{P: 3}
	tree := NewKDTree(data, n, dims, metric, 1)

	for k := 1; k <= n; k++ {
		indices, distances := tree.QueryKNN(data, n, k)
		for q := 0; q < n; q++ {
			bruteIdx, bruteDist := bruteForceKNN(data, n, dims, q, k, metric)
			if !knnResultsMatch(indices[q], distances[q], bruteIdx, bruteDist, floatTol) {
				t.Errorf("k=%d query=%d: tree KNN doesn't match brute force", k, q)
			}
		}
	}
}

func TestKDTree_KNN_AllSamePoints(t *testing.T) {
	// All 4 points are identical.
	data := []float64{5, 5, 5, 5, 5, 5, 5, 5}
	n, dims := 4, 2
	tree := NewKDTree(data, n, dims, EuclideanMetric{}, 2)

	indices, distances := tree.QueryKNN(data, n, 3)
	for q := 0; q < n; q++ {
		for j := 0; j < len(distances[q]); j++ {
			if distances[q][j] != 0 {
				t.Errorf("query %d: expected all distances 0, got %v", q, distances[q][j])
			}
		}
		if len(indices[q]) != 3 {
			t.Errorf("query %d: expected 3 results, got %d", q, len(indices[q]))
		}
	}
}

func TestKDTree_KNN_KEqualsN(t *testing.T) {
	data := []float64{0, 0, 1, 1, 2, 2}
	n, dims := 3, 2
	tree := NewKDTree(data, n, dims, EuclideanMetric{}, 1)

	indices, distances := tree.QueryKNN(data, n, n)
	for q := 0; q < n; q++ {
		if len(indices[q]) != n {
			t.Errorf("query %d: expected %d results, got %d", q, n, len(indices[q]))
		}
		// First distance should be 0 (self).
		if distances[q][0] != 0 {
			t.Errorf("query %d: expected self-distance 0, got %v", q, distances[q][0])
		}
	}
}

// --- MinRdistDual tests ---

func TestKDTree_MinRdistDual_SameNode(t *testing.T) {
	data := []float64{0, 0, 1, 1, 2, 2, 3, 3}
	tree := NewKDTree(data, 4, 2, EuclideanMetric{}, 2)

	// MinRdistDual of a node with itself should be 0.
	rdist := tree.MinRdistDual(0, 0)
	if rdist != 0 {
		t.Errorf("MinRdistDual(0, 0) = %v, want 0", rdist)
	}
}

func TestKDTree_MinRdistDual_LowerBound(t *testing.T) {
	data := []float64{
		0, 0,
		1, 0,
		10, 0,
		11, 0,
	}
	n, dims := 4, 2
	tree := NewKDTree(data, n, dims, EuclideanMetric{}, 2)

	// For all pairs of nodes, verify that MinRdistDual is a valid lower bound
	// on the actual reduced distance between any pair of points in the nodes.
	for i := 0; i < tree.NumNodes(); i++ {
		for j := 0; j < tree.NumNodes(); j++ {
			lb := tree.MinRdistDual(i, j)
			minActual := minRdistBetweenNodes(tree.data, tree.idxArray, tree.nodes, tree.dims, i, j, tree.metric)
			if lb > minActual+floatTol {
				t.Errorf("MinRdistDual(%d, %d) = %v > actual min rdist %v", i, j, lb, minActual)
			}
		}
	}
}

// --- MinRdistPoint tests ---

func TestKDTree_MinRdistPoint_LowerBound(t *testing.T) {
	data := []float64{
		0, 0,
		1, 1,
		5, 5,
		6, 6,
	}
	n, dims := 4, 2
	tree := NewKDTree(data, n, dims, EuclideanMetric{}, 2)

	testPoints := [][]float64{
		{3, 3},
		{-1, -1},
		{10, 10},
		{0, 0},
	}

	for _, pt := range testPoints {
		for nodeID := 0; nodeID < tree.NumNodes(); nodeID++ {
			lb := tree.MinRdistPoint(nodeID, pt)
			minActual := minRdistPointToNode(tree.data, tree.idxArray, tree.nodes, tree.dims, nodeID, pt, tree.metric)
			if lb > minActual+floatTol {
				t.Errorf("MinRdistPoint(%d, %v) = %v > actual %v", nodeID, pt, lb, minActual)
			}
		}
	}
}

// --- ChildNodes tests ---

func TestKDTree_ChildNodes(t *testing.T) {
	data := []float64{0, 0, 1, 1, 2, 2, 3, 3}
	tree := NewKDTree(data, 4, 2, EuclideanMetric{}, 1)

	left, right := tree.ChildNodes(0)
	if left != 1 || right != 2 {
		t.Errorf("ChildNodes(0) = (%d, %d), want (1, 2)", left, right)
	}
}

// --- Helper: brute-force KNN ---

func bruteForceKNN(data []float64, n, dims, queryIdx, k int, metric DistanceMetric) ([]int, []float64) {
	type distIdx struct {
		dist  float64
		index int
	}
	query := data[queryIdx*dims : (queryIdx+1)*dims]
	all := make([]distIdx, n)
	for i := 0; i < n; i++ {
		pt := data[i*dims : (i+1)*dims]
		all[i] = distIdx{dist: metric.Distance(query, pt), index: i}
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].dist == all[j].dist {
			return all[i].index < all[j].index
		}
		return all[i].dist < all[j].dist
	})
	if k > n {
		k = n
	}
	idx := make([]int, k)
	dists := make([]float64, k)
	for i := 0; i < k; i++ {
		idx[i] = all[i].index
		dists[i] = all[i].dist
	}
	return idx, dists
}

// knnResultsMatch checks that two KNN results agree on distances (indices
// may differ when distances are tied).
func knnResultsMatch(idx1 []int, dist1 []float64, idx2 []int, dist2 []float64, tol float64) bool {
	if len(dist1) != len(dist2) {
		return false
	}
	for i := range dist1 {
		if !almostEqual(dist1[i], dist2[i], tol) {
			return false
		}
	}
	return true
}

// minRdistBetweenNodes computes the actual minimum reduced distance between
// any pair of points in two tree nodes.
func minRdistBetweenNodes(data []float64, idxArray []int, nodes []NodeData, dims, node1, node2 int, metric DistanceMetric) float64 {
	if node1 >= len(nodes) || node2 >= len(nodes) {
		return math.Inf(1)
	}
	n1 := nodes[node1]
	n2 := nodes[node2]
	if n1.IdxEnd == 0 && n1.IdxStart == 0 && node1 != 0 {
		return math.Inf(1)
	}
	if n2.IdxEnd == 0 && n2.IdxStart == 0 && node2 != 0 {
		return math.Inf(1)
	}
	minRdist := math.Inf(1)
	for i := n1.IdxStart; i < n1.IdxEnd; i++ {
		pi := idxArray[i]
		ptI := data[pi*dims : (pi+1)*dims]
		for j := n2.IdxStart; j < n2.IdxEnd; j++ {
			pj := idxArray[j]
			ptJ := data[pj*dims : (pj+1)*dims]
			rd := metric.ReducedDistance(ptI, ptJ)
			if rd < minRdist {
				minRdist = rd
			}
		}
	}
	return minRdist
}

// minRdistPointToNode computes the actual minimum reduced distance from
// a point to any point in a tree node.
func minRdistPointToNode(data []float64, idxArray []int, nodes []NodeData, dims, nodeID int, point []float64, metric DistanceMetric) float64 {
	if nodeID >= len(nodes) {
		return math.Inf(1)
	}
	nd := nodes[nodeID]
	if nd.IdxEnd == 0 && nd.IdxStart == 0 && nodeID != 0 {
		return math.Inf(1)
	}
	minRdist := math.Inf(1)
	for i := nd.IdxStart; i < nd.IdxEnd; i++ {
		pi := idxArray[i]
		pt := data[pi*dims : (pi+1)*dims]
		rd := metric.ReducedDistance(point, pt)
		if rd < minRdist {
			minRdist = rd
		}
	}
	return minRdist
}
