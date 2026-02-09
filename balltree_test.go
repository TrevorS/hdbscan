package hdbscan

import (
	"math"
	"testing"
)

// --- Construction tests ---

func TestBallTree_Construction_BasicProperties(t *testing.T) {
	data := []float64{
		0, 0,
		1, 0,
		2, 0,
		0, 3,
		1, 3,
		2, 3,
	}
	n, dims := 6, 2
	tree := NewBallTree(data, n, dims, EuclideanMetric{}, 2)

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

func TestBallTree_Construction_LeafSize1(t *testing.T) {
	data := []float64{0, 0, 1, 1, 2, 2, 3, 3}
	tree := NewBallTree(data, 4, 2, EuclideanMetric{}, 1)

	for _, nd := range tree.NodeDataArray() {
		if nd.IsLeaf && (nd.IdxEnd-nd.IdxStart) != 1 {
			t.Errorf("leaf has %d points, want 1", nd.IdxEnd-nd.IdxStart)
		}
	}
}

func TestBallTree_Construction_LeafSizeLargerThanN(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	tree := NewBallTree(data, 2, 2, EuclideanMetric{}, 100)

	nodes := tree.NodeDataArray()
	if len(nodes) != 1 {
		t.Errorf("expected 1 node for leafSize > n, got %d", len(nodes))
	}
	if !nodes[0].IsLeaf {
		t.Error("root should be a leaf when leafSize > n")
	}
}

func TestBallTree_Construction_SinglePoint(t *testing.T) {
	data := []float64{5, 5}
	tree := NewBallTree(data, 1, 2, EuclideanMetric{}, 10)

	if tree.NumPoints() != 1 {
		t.Errorf("NumPoints() = %d, want 1", tree.NumPoints())
	}
	if tree.NumNodes() != 1 {
		t.Errorf("NumNodes() = %d, want 1", tree.NumNodes())
	}
}

func TestBallTree_Construction_TwoPoints(t *testing.T) {
	data := []float64{0, 0, 10, 10}
	tree := NewBallTree(data, 2, 2, EuclideanMetric{}, 1)

	if tree.NumPoints() != 2 {
		t.Errorf("NumPoints() = %d, want 2", tree.NumPoints())
	}
}

func TestBallTree_Construction_RadiusNonNegative(t *testing.T) {
	data := []float64{0, 0, 1, 1, 5, 5, 6, 6}
	tree := NewBallTree(data, 4, 2, EuclideanMetric{}, 2)

	for i, nd := range tree.NodeDataArray() {
		if nd.Radius < 0 {
			t.Errorf("node %d has negative radius %v", i, nd.Radius)
		}
	}
}

// --- KNN query tests ---

func TestBallTree_KNN_BruteForceMatch(t *testing.T) {
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
		tree := NewBallTree(data, n, dims, metric, 1)
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

func TestBallTree_KNN_Minkowski(t *testing.T) {
	data := []float64{
		0, 0,
		1, 0,
		0, 1,
		1, 1,
	}
	n, dims := 4, 2
	metric := MinkowskiMetric{P: 3}
	tree := NewBallTree(data, n, dims, metric, 1)

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

func TestBallTree_KNN_AllSamePoints(t *testing.T) {
	data := []float64{5, 5, 5, 5, 5, 5, 5, 5}
	n, dims := 4, 2
	tree := NewBallTree(data, n, dims, EuclideanMetric{}, 2)

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

func TestBallTree_KNN_KEqualsN(t *testing.T) {
	data := []float64{0, 0, 1, 1, 2, 2}
	n, dims := 3, 2
	tree := NewBallTree(data, n, dims, EuclideanMetric{}, 1)

	indices, distances := tree.QueryKNN(data, n, n)
	for q := 0; q < n; q++ {
		if len(indices[q]) != n {
			t.Errorf("query %d: expected %d results, got %d", q, n, len(indices[q]))
		}
		if distances[q][0] != 0 {
			t.Errorf("query %d: expected self-distance 0, got %v", q, distances[q][0])
		}
	}
}

// --- MinRdistDual tests ---

func TestBallTree_MinRdistDual_SameNode(t *testing.T) {
	data := []float64{0, 0, 1, 1, 2, 2, 3, 3}
	tree := NewBallTree(data, 4, 2, EuclideanMetric{}, 2)

	rdist := tree.MinRdistDual(0, 0)
	if rdist != 0 {
		t.Errorf("MinRdistDual(0, 0) = %v, want 0", rdist)
	}
}

func TestBallTree_MinRdistDual_LowerBound(t *testing.T) {
	data := []float64{
		0, 0,
		1, 0,
		10, 0,
		11, 0,
	}
	n, dims := 4, 2
	tree := NewBallTree(data, n, dims, EuclideanMetric{}, 2)

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

func TestBallTree_MinRdistPoint_LowerBound(t *testing.T) {
	data := []float64{
		0, 0,
		1, 1,
		5, 5,
		6, 6,
	}
	n, dims := 4, 2
	tree := NewBallTree(data, n, dims, EuclideanMetric{}, 2)

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

func TestBallTree_ChildNodes(t *testing.T) {
	data := []float64{0, 0, 1, 1, 2, 2, 3, 3}
	tree := NewBallTree(data, 4, 2, EuclideanMetric{}, 1)

	left, right := tree.ChildNodes(0)
	if left != 1 || right != 2 {
		t.Errorf("ChildNodes(0) = (%d, %d), want (1, 2)", left, right)
	}
}

// --- Larger dataset stress test ---

func TestBallTree_KNN_LargerDataset(t *testing.T) {
	// Generate a grid of 25 points in 2D.
	n, dims := 25, 2
	data := make([]float64, n*dims)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			idx := i*5 + j
			data[idx*dims] = float64(i)
			data[idx*dims+1] = float64(j)
		}
	}

	tree := NewBallTree(data, n, dims, EuclideanMetric{}, 3)

	// Verify KNN for all points matches brute force.
	for k := 1; k <= 5; k++ {
		indices, distances := tree.QueryKNN(data, n, k)
		for q := 0; q < n; q++ {
			_, bruteDist := bruteForceKNN(data, n, dims, q, k, EuclideanMetric{})
			if !knnResultsMatch(indices[q], distances[q], nil, bruteDist, floatTol) {
				t.Errorf("k=%d query=%d: distances don't match brute force", k, q)
			}
		}
	}
}

// --- Interface compliance checks ---

func TestKDTree_ImplementsSpatialTree(t *testing.T) {
	var _ SpatialTree = (*KDTree)(nil)
}

func TestKDTree_ImplementsBoruvkaTree(t *testing.T) {
	var _ BoruvkaTree = (*KDTree)(nil)
}

func TestBallTree_ImplementsSpatialTree(t *testing.T) {
	var _ SpatialTree = (*BallTree)(nil)
}

func TestBallTree_ImplementsBoruvkaTree(t *testing.T) {
	var _ BoruvkaTree = (*BallTree)(nil)
}

// --- MinRdistDual with different metrics ---

func TestKDTree_MinRdistDual_Manhattan(t *testing.T) {
	data := []float64{
		0, 0,
		1, 0,
		10, 0,
		11, 0,
	}
	n, dims := 4, 2
	metric := ManhattanMetric{}
	tree := NewKDTree(data, n, dims, metric, 2)

	for i := 0; i < tree.NumNodes(); i++ {
		for j := 0; j < tree.NumNodes(); j++ {
			lb := tree.MinRdistDual(i, j)
			minActual := minRdistBetweenNodes(tree.data, tree.idxArray, tree.nodes, tree.dims, i, j, metric)
			if lb > minActual+floatTol {
				t.Errorf("MinRdistDual(%d, %d) = %v > actual %v", i, j, lb, minActual)
			}
		}
	}
}

func TestKDTree_MinRdistDual_Chebyshev(t *testing.T) {
	data := []float64{
		0, 0,
		1, 1,
		10, 10,
		11, 11,
	}
	n, dims := 4, 2
	metric := ChebyshevMetric{}
	tree := NewKDTree(data, n, dims, metric, 2)

	for i := 0; i < tree.NumNodes(); i++ {
		for j := 0; j < tree.NumNodes(); j++ {
			lb := tree.MinRdistDual(i, j)
			minActual := minRdistBetweenNodes(tree.data, tree.idxArray, tree.nodes, tree.dims, i, j, metric)
			if lb > minActual+floatTol {
				t.Errorf("MinRdistDual(%d, %d) = %v > actual %v", i, j, lb, minActual)
			}
		}
	}
}

func TestBallTree_MinRdistDual_Manhattan(t *testing.T) {
	data := []float64{
		0, 0,
		1, 0,
		10, 0,
		11, 0,
	}
	n, dims := 4, 2
	metric := ManhattanMetric{}
	tree := NewBallTree(data, n, dims, metric, 2)

	for i := 0; i < tree.NumNodes(); i++ {
		for j := 0; j < tree.NumNodes(); j++ {
			lb := tree.MinRdistDual(i, j)
			minActual := minRdistBetweenNodes(tree.data, tree.idxArray, tree.nodes, tree.dims, i, j, metric)
			if lb > minActual+floatTol {
				t.Errorf("MinRdistDual(%d, %d) = %v > actual %v (centroidDist=%v, r1=%v, r2=%v)",
					i, j, lb, minActual,
					tree.centroidDists[i*tree.numNodesAlloc+j],
					tree.nodes[i].Radius, tree.nodes[j].Radius)
			}
		}
	}
}

// --- Edge case: MinRdistPoint for points inside the bounding box ---

func TestKDTree_MinRdistPoint_PointInsideBox(t *testing.T) {
	data := []float64{0, 0, 10, 10}
	tree := NewKDTree(data, 2, 2, EuclideanMetric{}, 10)

	// Point inside the root's bounding box should have rdist = 0.
	rdist := tree.MinRdistPoint(0, []float64{5, 5})
	if rdist != 0 {
		t.Errorf("point inside box: MinRdistPoint = %v, want 0", rdist)
	}
}

func TestBallTree_MinRdistPoint_PointInsideBall(t *testing.T) {
	data := []float64{0, 0, 2, 0}
	tree := NewBallTree(data, 2, 2, EuclideanMetric{}, 10)

	// Centroid is at (1, 0), radius covers both points.
	// Point at (1, 0) is exactly the centroid.
	rdist := tree.MinRdistPoint(0, []float64{1, 0})
	if rdist != 0 {
		t.Errorf("point at centroid: MinRdistPoint = %v, want 0", rdist)
	}
}

// --- Verify that nodes have non-overlapping point ranges at leaf level ---

func TestKDTree_LeafPointsCoverAll(t *testing.T) {
	data := make([]float64, 20*3)
	for i := range data {
		data[i] = float64(i)
	}
	n, dims := 20, 3
	tree := NewKDTree(data, n, dims, EuclideanMetric{}, 4)

	covered := make([]bool, n)
	for _, nd := range tree.NodeDataArray() {
		if nd.IsLeaf {
			for i := nd.IdxStart; i < nd.IdxEnd; i++ {
				origIdx := tree.idxArray[i]
				if covered[origIdx] {
					t.Errorf("point %d appears in multiple leaves", origIdx)
				}
				covered[origIdx] = true
			}
		}
	}
	for i, c := range covered {
		if !c {
			t.Errorf("point %d not covered by any leaf", i)
		}
	}
}

func TestBallTree_LeafPointsCoverAll(t *testing.T) {
	data := make([]float64, 20*3)
	for i := range data {
		data[i] = float64(i)
	}
	n, dims := 20, 3
	tree := NewBallTree(data, n, dims, EuclideanMetric{}, 4)

	covered := make([]bool, n)
	for _, nd := range tree.NodeDataArray() {
		if nd.IsLeaf {
			for i := nd.IdxStart; i < nd.IdxEnd; i++ {
				origIdx := tree.idxArray[i]
				if covered[origIdx] {
					t.Errorf("point %d appears in multiple leaves", origIdx)
				}
				covered[origIdx] = true
			}
		}
	}
	for i, c := range covered {
		if !c {
			t.Errorf("point %d not covered by any leaf", i)
		}
	}
}

// --- KDTree larger dataset KNN test ---

func TestKDTree_KNN_LargerDataset(t *testing.T) {
	n, dims := 25, 2
	data := make([]float64, n*dims)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			idx := i*5 + j
			data[idx*dims] = float64(i)
			data[idx*dims+1] = float64(j)
		}
	}

	tree := NewKDTree(data, n, dims, EuclideanMetric{}, 3)

	for k := 1; k <= 5; k++ {
		indices, distances := tree.QueryKNN(data, n, k)
		for q := 0; q < n; q++ {
			_, bruteDist := bruteForceKNN(data, n, dims, q, k, EuclideanMetric{})
			if !knnResultsMatch(indices[q], distances[q], nil, bruteDist, floatTol) {
				t.Errorf("k=%d query=%d: distances don't match brute force.\n  tree: %v\n  brute: %v",
					k, q, distances[q], bruteDist)
			}
		}
	}
}

// --- Empty tree ---

func TestKDTree_EmptyData(t *testing.T) {
	tree := NewKDTree(nil, 0, 2, EuclideanMetric{}, 10)
	if tree.NumPoints() != 0 {
		t.Errorf("NumPoints() = %d, want 0", tree.NumPoints())
	}
}

func TestBallTree_EmptyData(t *testing.T) {
	tree := NewBallTree(nil, 0, 2, EuclideanMetric{}, 10)
	if tree.NumPoints() != 0 {
		t.Errorf("NumPoints() = %d, want 0", tree.NumPoints())
	}
}

// --- Higher dimensionality ---

func TestKDTree_KNN_HigherDim(t *testing.T) {
	// 10 points in 5D.
	n, dims := 10, 5
	data := make([]float64, n*dims)
	for i := range data {
		data[i] = float64(i) * 0.7
	}

	tree := NewKDTree(data, n, dims, EuclideanMetric{}, 2)
	for k := 1; k <= n; k++ {
		_, distances := tree.QueryKNN(data, n, k)
		for q := 0; q < n; q++ {
			_, bruteDist := bruteForceKNN(data, n, dims, q, k, EuclideanMetric{})
			if !knnResultsMatch(nil, distances[q], nil, bruteDist, floatTol) {
				t.Errorf("k=%d query=%d: distances don't match brute force in 5D", k, q)
			}
		}
	}
}

func TestBallTree_KNN_HigherDim(t *testing.T) {
	n, dims := 10, 5
	data := make([]float64, n*dims)
	for i := range data {
		data[i] = float64(i) * 0.7
	}

	tree := NewBallTree(data, n, dims, EuclideanMetric{}, 2)
	for k := 1; k <= n; k++ {
		_, distances := tree.QueryKNN(data, n, k)
		for q := 0; q < n; q++ {
			_, bruteDist := bruteForceKNN(data, n, dims, q, k, EuclideanMetric{})
			if !knnResultsMatch(nil, distances[q], nil, bruteDist, floatTol) {
				t.Errorf("k=%d query=%d: distances don't match brute force in 5D", k, q)
			}
		}
	}
}

// --- Ensure no NaN/Inf in normal operation ---

func TestKDTree_NoNaNInf(t *testing.T) {
	data := []float64{0, 0, 1, 0, 0, 1, 1, 1, 0.5, 0.5}
	n, dims := 5, 2
	tree := NewKDTree(data, n, dims, EuclideanMetric{}, 2)

	_, distances := tree.QueryKNN(data, n, 3)
	for q := 0; q < n; q++ {
		for _, d := range distances[q] {
			if math.IsNaN(d) || math.IsInf(d, 0) {
				t.Errorf("query %d: got NaN or Inf distance %v", q, d)
			}
		}
	}
}

func TestBallTree_NoNaNInf(t *testing.T) {
	data := []float64{0, 0, 1, 0, 0, 1, 1, 1, 0.5, 0.5}
	n, dims := 5, 2
	tree := NewBallTree(data, n, dims, EuclideanMetric{}, 2)

	_, distances := tree.QueryKNN(data, n, 3)
	for q := 0; q < n; q++ {
		for _, d := range distances[q] {
			if math.IsNaN(d) || math.IsInf(d, 0) {
				t.Errorf("query %d: got NaN or Inf distance %v", q, d)
			}
		}
	}
}
