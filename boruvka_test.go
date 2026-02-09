package hdbscan

import (
	"math"
	"sort"
	"testing"
)

// bruteForceBoruvkaTree is a mock BoruvkaTree that stores all points in a
// simple binary tree structure (one point per leaf, or a few per leaf).
// It implements the full BoruvkaTree interface using brute-force computation,
// allowing us to test the Borůvka algorithm without real KD-tree/Ball tree.
type bruteForceBoruvkaTree struct {
	data     []float64
	n        int
	dims     int
	metric   DistanceMetric
	nodes    []NodeData
	idxArray []int
	leafSize int
}

// newBruteForceBoruvkaTree builds a simple binary tree over the data.
// Data is flat row-major with n rows and dims columns.
func newBruteForceBoruvkaTree(data []float64, n, dims, leafSize int, metric DistanceMetric) *bruteForceBoruvkaTree {
	t := &bruteForceBoruvkaTree{
		data:     make([]float64, len(data)),
		n:        n,
		dims:     dims,
		metric:   metric,
		leafSize: leafSize,
	}
	copy(t.data, data)

	// Identity permutation — no reordering.
	t.idxArray = make([]int, n)
	for i := range t.idxArray {
		t.idxArray[i] = i
	}

	// Build a complete binary tree. We pre-allocate enough nodes.
	t.nodes = nil
	t.buildNode(0, n)

	return t
}

func (t *bruteForceBoruvkaTree) buildNode(start, end int) int {
	idx := len(t.nodes)
	count := end - start

	if count <= t.leafSize {
		t.nodes = append(t.nodes, NodeData{
			IdxStart: start,
			IdxEnd:   end,
			IsLeaf:   true,
			Radius:   t.computeRadius(start, end),
		})
		return idx
	}

	// Placeholder — we'll fill in after building children.
	t.nodes = append(t.nodes, NodeData{})

	mid := (start + end) / 2
	leftIdx := t.buildNode(start, mid)
	rightIdx := t.buildNode(mid, end)

	_ = leftIdx
	_ = rightIdx

	t.nodes[idx] = NodeData{
		IdxStart: start,
		IdxEnd:   end,
		IsLeaf:   false,
		Radius:   t.computeRadius(start, end),
	}

	return idx
}

func (t *bruteForceBoruvkaTree) computeRadius(start, end int) float64 {
	if end-start <= 1 {
		return 0
	}
	// Compute centroid.
	centroid := make([]float64, t.dims)
	count := end - start
	for i := start; i < end; i++ {
		p := t.idxArray[i]
		for d := 0; d < t.dims; d++ {
			centroid[d] += t.data[p*t.dims+d]
		}
	}
	for d := range centroid {
		centroid[d] /= float64(count)
	}
	// Max distance from centroid.
	maxDist := 0.0
	for i := start; i < end; i++ {
		p := t.idxArray[i]
		d := t.metric.Distance(t.data[p*t.dims:(p+1)*t.dims], centroid)
		if d > maxDist {
			maxDist = d
		}
	}
	return maxDist
}

func (t *bruteForceBoruvkaTree) Data() []float64           { return t.data }
func (t *bruteForceBoruvkaTree) NumPoints() int            { return t.n }
func (t *bruteForceBoruvkaTree) NumFeatures() int          { return t.dims }
func (t *bruteForceBoruvkaTree) IdxArray() []int           { return t.idxArray }
func (t *bruteForceBoruvkaTree) NodeDataArray() []NodeData { return t.nodes }
func (t *bruteForceBoruvkaTree) NumNodes() int             { return len(t.nodes) }

func (t *bruteForceBoruvkaTree) ChildNodes(node int) (left, right int) {
	// In our tree structure, children are the next two nodes built after the parent.
	// We need to figure out which nodes are children. Since we build recursively,
	// the left child is always at node+1. The right child starts after the left subtree.
	// We'll find it by counting nodes in the left subtree.
	nd := t.nodes[node]
	if nd.IsLeaf {
		return -1, -1
	}
	left = node + 1
	// The right child starts after all the nodes in the left subtree.
	// Left subtree covers [nd.IdxStart, mid) and right covers [mid, nd.IdxEnd).
	right = left + t.subtreeSize(left)
	return left, right
}

func (t *bruteForceBoruvkaTree) subtreeSize(node int) int {
	if t.nodes[node].IsLeaf {
		return 1
	}
	left := node + 1
	leftSize := t.subtreeSize(left)
	right := left + leftSize
	rightSize := t.subtreeSize(right)
	return 1 + leftSize + rightSize
}

func (t *bruteForceBoruvkaTree) MinRdistDual(node1, node2 int) float64 {
	// Brute force: minimum reduced distance between any point in node1 and node2.
	nd1 := t.nodes[node1]
	nd2 := t.nodes[node2]

	minD := math.MaxFloat64
	for i := nd1.IdxStart; i < nd1.IdxEnd; i++ {
		p := t.idxArray[i]
		for j := nd2.IdxStart; j < nd2.IdxEnd; j++ {
			q := t.idxArray[j]
			if p == q {
				continue
			}
			d := t.metric.ReducedDistance(
				t.data[p*t.dims:(p+1)*t.dims],
				t.data[q*t.dims:(q+1)*t.dims],
			)
			if d < minD {
				minD = d
			}
		}
	}
	if minD == math.MaxFloat64 {
		return 0
	}
	return minD
}

func (t *bruteForceBoruvkaTree) MinRdistPoint(node int, point []float64) float64 {
	nd := t.nodes[node]
	minD := math.MaxFloat64
	for i := nd.IdxStart; i < nd.IdxEnd; i++ {
		p := t.idxArray[i]
		d := t.metric.ReducedDistance(t.data[p*t.dims:(p+1)*t.dims], point)
		if d < minD {
			minD = d
		}
	}
	return minD
}

func (t *bruteForceBoruvkaTree) QueryKNN(queryData []float64, queryRows, k int) ([][]int, [][]float64) {
	indices := make([][]int, queryRows)
	distances := make([][]float64, queryRows)

	for i := 0; i < queryRows; i++ {
		q := queryData[i*t.dims : (i+1)*t.dims]

		type nd struct {
			idx  int
			dist float64
		}
		var dists []nd
		for j := 0; j < t.n; j++ {
			d := t.metric.Distance(q, t.data[j*t.dims:(j+1)*t.dims])
			dists = append(dists, nd{j, d})
		}
		sort.Slice(dists, func(a, b int) bool { return dists[a].dist < dists[b].dist })

		kk := k
		if kk > len(dists) {
			kk = len(dists)
		}
		idx := make([]int, kk)
		dist := make([]float64, kk)
		for j := 0; j < kk; j++ {
			idx[j] = dists[j].idx
			dist[j] = dists[j].dist
		}
		indices[i] = idx
		distances[i] = dist
	}
	return indices, distances
}

// computeBruteForceMST computes the MST using Prim's on the full mutual
// reachability matrix, for comparison with Borůvka results.
func computeBruteForceMST(data []float64, n, dims int, metric DistanceMetric, minSamples int, alpha float64) (float64, [][3]float64) {
	distMatrix := ComputePairwiseDistances(data, n, dims, metric)
	ms := minSamples
	if ms > n-1 {
		ms = n - 1
	}
	coreDistances := ComputeCoreDistances(distMatrix, n, ms)
	mrMatrix := MutualReachability(distMatrix, coreDistances, n, alpha)
	edges := PrimMST(mrMatrix, n)

	totalWeight := 0.0
	for _, e := range edges {
		totalWeight += e[2]
	}
	return totalWeight, edges
}

func boruvkaTotalMSTWeight(edges [][3]float64) float64 {
	w := 0.0
	for _, e := range edges {
		w += e[2]
	}
	return w
}

func TestBoruvkaTotalWeightMatchesBrute(t *testing.T) {
	// 6-point dataset with distinct distances.
	data := []float64{
		0, 0,
		1, 0,
		5, 5,
		6, 5,
		10, 0,
		10, 1,
	}
	n := 6
	dims := 2
	metric := EuclideanMetric{}
	minSamples := 2
	alpha := 1.0

	bruteWeight, _ := computeBruteForceMST(data, n, dims, metric, minSamples, alpha)

	tree := newBruteForceBoruvkaTree(data, n, dims, 2, metric)
	boruvka := NewKDTreeBoruvka(tree, metric, minSamples, alpha)
	edges, coreDist := boruvka.SpanningTree()

	if len(edges) != n-1 {
		t.Fatalf("expected %d edges, got %d", n-1, len(edges))
	}
	if len(coreDist) != n {
		t.Fatalf("expected %d core distances, got %d", n, len(coreDist))
	}

	boruvkaWeight := boruvkaTotalMSTWeight(edges)
	if math.Abs(boruvkaWeight-bruteWeight) > 1e-9 {
		t.Errorf("MST total weight mismatch: brute=%f, boruvka=%f", bruteWeight, boruvkaWeight)
	}
}

func TestBoruvkaBallTreeTotalWeightMatchesBrute(t *testing.T) {
	data := []float64{
		0, 0,
		1, 0,
		5, 5,
		6, 5,
		10, 0,
		10, 1,
	}
	n := 6
	dims := 2
	metric := EuclideanMetric{}
	minSamples := 2
	alpha := 1.0

	bruteWeight, _ := computeBruteForceMST(data, n, dims, metric, minSamples, alpha)

	tree := newBruteForceBoruvkaTree(data, n, dims, 2, metric)
	boruvka := NewBallTreeBoruvka(tree, metric, minSamples, alpha)
	edges, _ := boruvka.SpanningTree()

	if len(edges) != n-1 {
		t.Fatalf("expected %d edges, got %d", n-1, len(edges))
	}

	boruvkaWeight := boruvkaTotalMSTWeight(edges)
	if math.Abs(boruvkaWeight-bruteWeight) > 1e-9 {
		t.Errorf("MST total weight mismatch: brute=%f, boruvka=%f", bruteWeight, boruvkaWeight)
	}
}

func TestBoruvkaManhattan(t *testing.T) {
	data := []float64{
		0, 0,
		1, 1,
		3, 3,
		4, 4,
		8, 8,
		9, 9,
	}
	n := 6
	dims := 2
	metric := ManhattanMetric{}
	minSamples := 2
	alpha := 1.0

	bruteWeight, _ := computeBruteForceMST(data, n, dims, metric, minSamples, alpha)

	tree := newBruteForceBoruvkaTree(data, n, dims, 2, metric)
	boruvka := NewKDTreeBoruvka(tree, metric, minSamples, alpha)
	edges, _ := boruvka.SpanningTree()

	if len(edges) != n-1 {
		t.Fatalf("expected %d edges, got %d", n-1, len(edges))
	}
	boruvkaWeight := boruvkaTotalMSTWeight(edges)
	if math.Abs(boruvkaWeight-bruteWeight) > 1e-9 {
		t.Errorf("MST total weight mismatch: brute=%f, boruvka=%f", bruteWeight, boruvkaWeight)
	}
}

func TestBoruvkaMinSamples(t *testing.T) {
	data := []float64{
		0, 0,
		1, 0,
		5, 5,
		6, 5,
		10, 0,
		10, 1,
	}
	n := 6
	dims := 2
	metric := EuclideanMetric{}
	alpha := 1.0

	for _, ms := range []int{1, 3, 5} {
		t.Run("", func(t *testing.T) {
			bruteWeight, _ := computeBruteForceMST(data, n, dims, metric, ms, alpha)

			tree := newBruteForceBoruvkaTree(data, n, dims, 2, metric)
			boruvka := NewKDTreeBoruvka(tree, metric, ms, alpha)
			edges, _ := boruvka.SpanningTree()

			if len(edges) != n-1 {
				t.Fatalf("expected %d edges, got %d", n-1, len(edges))
			}
			boruvkaWeight := boruvkaTotalMSTWeight(edges)
			if math.Abs(boruvkaWeight-bruteWeight) > 1e-9 {
				t.Errorf("minSamples=%d: MST total weight mismatch: brute=%f, boruvka=%f", ms, bruteWeight, boruvkaWeight)
			}
		})
	}
}

func TestBoruvkaTwoPoints(t *testing.T) {
	data := []float64{0, 0, 3, 4}
	n := 2
	dims := 2
	metric := EuclideanMetric{}

	tree := newBruteForceBoruvkaTree(data, n, dims, 2, metric)
	boruvka := NewKDTreeBoruvka(tree, metric, 1, 1.0)
	edges, coreDist := boruvka.SpanningTree()

	if len(edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(edges))
	}
	// Distance is 5.0; core distance with minSamples=1 is 5.0; mr = max(5,5,5) = 5.
	if math.Abs(edges[0][2]-5.0) > 1e-9 {
		t.Errorf("expected edge weight 5.0, got %f", edges[0][2])
	}
	if len(coreDist) != 2 {
		t.Fatalf("expected 2 core distances, got %d", len(coreDist))
	}
	for i, cd := range coreDist {
		if math.Abs(cd-5.0) > 1e-9 {
			t.Errorf("core distance[%d] = %f, expected 5.0", i, cd)
		}
	}
}

func TestBoruvkaAlpha(t *testing.T) {
	// With alpha != 1.0, distances get scaled.
	data := []float64{
		0, 0,
		1, 0,
		5, 5,
		6, 5,
		10, 0,
		10, 1,
	}
	n := 6
	dims := 2
	metric := EuclideanMetric{}
	minSamples := 2
	alpha := 0.5

	bruteWeight, _ := computeBruteForceMST(data, n, dims, metric, minSamples, alpha)

	tree := newBruteForceBoruvkaTree(data, n, dims, 2, metric)
	boruvka := NewKDTreeBoruvka(tree, metric, minSamples, alpha)
	edges, _ := boruvka.SpanningTree()

	if len(edges) != n-1 {
		t.Fatalf("expected %d edges, got %d", n-1, len(edges))
	}
	boruvkaWeight := boruvkaTotalMSTWeight(edges)
	if math.Abs(boruvkaWeight-bruteWeight) > 1e-9 {
		t.Errorf("MST total weight mismatch: brute=%f, boruvka=%f", bruteWeight, boruvkaWeight)
	}
}

func TestBoruvkaUnionFind(t *testing.T) {
	uf := newBoruvkaUnionFind(5)

	// Initially 5 components.
	if got := len(uf.components()); got != 5 {
		t.Fatalf("expected 5 components, got %d", got)
	}

	uf.union(0, 1)
	if got := len(uf.components()); got != 4 {
		t.Fatalf("expected 4 components, got %d", got)
	}
	if uf.find(0) != uf.find(1) {
		t.Error("0 and 1 should be in same component")
	}

	uf.union(2, 3)
	uf.union(0, 3)
	if got := len(uf.components()); got != 2 {
		t.Fatalf("expected 2 components, got %d", got)
	}
	if uf.find(0) != uf.find(2) {
		t.Error("0 and 2 should be in same component after transitive union")
	}

	uf.union(0, 4)
	if got := len(uf.components()); got != 1 {
		t.Fatalf("expected 1 component, got %d", got)
	}
}

func TestBoruvkaFullPipeline(t *testing.T) {
	// Verify that the MST from Borůvka feeds correctly into the rest
	// of the HDBSCAN pipeline by comparing labels to brute force.
	data := []float64{
		0, 0,
		0.1, 0,
		0, 0.1,
		0.1, 0.1,
		0.05, 0.05,
		5, 5,
		5.1, 5,
		5, 5.1,
		5.1, 5.1,
		5.05, 5.05,
	}
	n := 10
	dims := 2
	metric := EuclideanMetric{}
	minSamples := 3

	// Brute force pipeline.
	distMatrix := ComputePairwiseDistances(data, n, dims, metric)
	coreDistBrute := ComputeCoreDistances(distMatrix, n, minSamples)
	mrMatrix := MutualReachability(distMatrix, coreDistBrute, n, 1.0)
	bruteEdges := PrimMST(mrMatrix, n)
	bruteDendrogram := Label(bruteEdges, n)
	bruteCondensed := CondenseTree(bruteDendrogram, 3)
	if bruteCondensed == nil {
		t.Fatal("brute-force condensed tree is nil")
	}
	bruteStab := ComputeStability(bruteCondensed)
	bruteSelected, _ := SelectClustersEOM(bruteCondensed, bruteStab, false, 0, math.Inf(1))
	bruteLabels, _ := GetLabelsAndProbabilities(bruteCondensed, bruteSelected, n, false, 0, false)

	// Borůvka pipeline.
	tree := newBruteForceBoruvkaTree(data, n, dims, 2, metric)
	boruvka := NewKDTreeBoruvka(tree, metric, minSamples, 1.0)
	boruvkaEdges, _ := boruvka.SpanningTree()
	boruvkaDendrogram := Label(boruvkaEdges, n)
	boruvkaCondensed := CondenseTree(boruvkaDendrogram, 3)
	if boruvkaCondensed == nil {
		t.Fatal("boruvka condensed tree is nil")
	}
	boruvkaStab := ComputeStability(boruvkaCondensed)
	boruvkaSelected, _ := SelectClustersEOM(boruvkaCondensed, boruvkaStab, false, 0, math.Inf(1))
	boruvkaLabels, _ := GetLabelsAndProbabilities(boruvkaCondensed, boruvkaSelected, n, false, 0, false)

	// Compare labels permutation-invariantly.
	if !boruvkaLabelsEquivalent(bruteLabels, boruvkaLabels) {
		t.Errorf("label mismatch:\n  brute:   %v\n  boruvka: %v", bruteLabels, boruvkaLabels)
	}
}

// boruvkaLabelsEquivalent checks if two label assignments are the same up to
// relabeling of cluster IDs (noise=-1 must match exactly).
func boruvkaLabelsEquivalent(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	aToB := map[int]int{}
	bToA := map[int]int{}
	for i := range a {
		if a[i] == -1 && b[i] == -1 {
			continue
		}
		if a[i] == -1 || b[i] == -1 {
			return false
		}
		if mapped, ok := aToB[a[i]]; ok {
			if mapped != b[i] {
				return false
			}
		} else {
			aToB[a[i]] = b[i]
		}
		if mapped, ok := bToA[b[i]]; ok {
			if mapped != a[i] {
				return false
			}
		} else {
			bToA[b[i]] = a[i]
		}
	}
	return true
}
