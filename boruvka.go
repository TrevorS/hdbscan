package hdbscan

import "math"

// boruvkaUnionFind is a lightweight union-find for Borůvka MST construction.
// It uses union by rank and path compression (halving).
type boruvkaUnionFind struct {
	parent      []int
	rank        []int
	isComponent []bool // true if this element is a component root
}

func newBoruvkaUnionFind(n int) *boruvkaUnionFind {
	parent := make([]int, n)
	rank := make([]int, n)
	isComp := make([]bool, n)
	for i := range parent {
		parent[i] = i
		isComp[i] = true
	}
	return &boruvkaUnionFind{parent: parent, rank: rank, isComponent: isComp}
}

func (uf *boruvkaUnionFind) find(x int) int {
	// Path halving: every other node points to its grandparent.
	for uf.parent[x] != x {
		uf.parent[x] = uf.parent[uf.parent[x]]
		x = uf.parent[x]
	}
	return x
}

func (uf *boruvkaUnionFind) union(x, y int) {
	xr := uf.find(x)
	yr := uf.find(y)
	if xr == yr {
		return
	}
	if uf.rank[xr] < uf.rank[yr] {
		uf.parent[xr] = yr
		uf.isComponent[xr] = false
	} else if uf.rank[xr] > uf.rank[yr] {
		uf.parent[yr] = xr
		uf.isComponent[yr] = false
	} else {
		uf.parent[yr] = xr
		uf.isComponent[yr] = false
		uf.rank[xr]++
	}
}

// components returns the list of current component root indices.
func (uf *boruvkaUnionFind) components() []int {
	var out []int
	for i, v := range uf.isComponent {
		if v {
			out = append(out, i)
		}
	}
	return out
}

// boruvkaState holds the shared state for dual-tree Borůvka MST construction.
// All distances (core, candidate, bounds) are stored in true distance space.
// The tree's MinRdistDual returns reduced distance, which we convert to true
// distance before comparing with bounds.
type boruvkaState struct {
	tree       BoruvkaTree
	metric     DistanceMetric
	alpha      float64
	minSamples int

	numPoints   int
	numFeatures int
	numNodes    int

	// Per-point data (indexed by original point index).
	// All distances are in true distance space.
	coreDistance      []float64
	componentOfPoint  []int
	candidateNeighbor []int
	candidatePoint    []int
	candidateDist     []float64

	// Per-node data.
	componentOfNode []int
	bounds          []float64 // true distance space

	uf         *boruvkaUnionFind
	components []int

	edges    [][3]float64
	numEdges int
}

func newBoruvkaState(tree BoruvkaTree, metric DistanceMetric, minSamples int, alpha float64) *boruvkaState {
	n := tree.NumPoints()
	dims := tree.NumFeatures()
	numNodes := tree.NumNodes()

	s := &boruvkaState{
		tree:        tree,
		metric:      metric,
		alpha:       alpha,
		minSamples:  minSamples,
		numPoints:   n,
		numFeatures: dims,
		numNodes:    numNodes,

		coreDistance:      make([]float64, n),
		componentOfPoint:  make([]int, n),
		candidateNeighbor: make([]int, n),
		candidatePoint:    make([]int, n),
		candidateDist:     make([]float64, n),
		componentOfNode:   make([]int, numNodes),
		bounds:            make([]float64, numNodes),

		uf:    newBoruvkaUnionFind(n),
		edges: make([][3]float64, 0, n-1),
	}

	s.initializeComponents()
	s.computeBounds()

	return s
}

func (s *boruvkaState) initializeComponents() {
	for i := 0; i < s.numPoints; i++ {
		s.componentOfPoint[i] = i
		s.candidateNeighbor[i] = -1
		s.candidatePoint[i] = -1
		s.candidateDist[i] = math.MaxFloat64
	}
	for i := 0; i < s.numNodes; i++ {
		s.componentOfNode[i] = -(i + 1) // negative = mixed/unknown
	}
}

func (s *boruvkaState) computeBounds() {
	n := s.numPoints
	ms := s.minSamples
	if ms > n-1 {
		ms = n - 1
	}

	// Compute core distances via tree KNN. QueryKNN returns true distances.
	data := s.tree.Data()
	knnIdx, knnDist := s.tree.QueryKNN(data, n, ms+1)

	// Store the ms-th nearest neighbor distance (true distance).
	for i := 0; i < n; i++ {
		s.coreDistance[i] = knnDist[i][ms]
	}

	// Quick first pass: use KNN results to seed initial candidates.
	// For each point, compute actual MR distance to its nearest neighbors
	// and use the best one as the initial candidate.
	for i := 0; i < n; i++ {
		for k := 0; k < len(knnIdx[i]); k++ {
			m := knnIdx[i][k]
			if m == i {
				continue
			}
			// Compute the actual MR distance.
			d := knnDist[i][k]
			if s.alpha != 1.0 {
				d /= s.alpha
			}
			mrDist := d
			if s.coreDistance[i] > mrDist {
				mrDist = s.coreDistance[i]
			}
			if s.coreDistance[m] > mrDist {
				mrDist = s.coreDistance[m]
			}
			if mrDist < s.candidateDist[i] {
				s.candidatePoint[i] = i
				s.candidateNeighbor[i] = m
				s.candidateDist[i] = mrDist
			}
		}
	}

	s.updateComponents()

	for i := 0; i < s.numNodes; i++ {
		s.bounds[i] = math.MaxFloat64
	}
}

func (s *boruvkaState) updateComponents() int {
	comps := s.uf.components()

	// For each component, try to add its best candidate edge.
	for _, component := range comps {
		source := s.candidatePoint[component]
		sink := s.candidateNeighbor[component]
		if source == -1 || sink == -1 {
			continue
		}
		srcComp := s.uf.find(source)
		sinkComp := s.uf.find(sink)
		if srcComp == sinkComp {
			s.candidatePoint[component] = -1
			s.candidateNeighbor[component] = -1
			s.candidateDist[component] = math.MaxFloat64
			continue
		}

		s.edges = append(s.edges, [3]float64{float64(source), float64(sink), s.candidateDist[component]})
		s.numEdges++

		s.uf.union(source, sink)
		s.candidateDist[component] = math.MaxFloat64

		if s.numEdges == s.numPoints-1 {
			s.components = s.uf.components()
			return len(s.components)
		}
	}

	// Propagate union-find results to componentOfPoint.
	for i := 0; i < s.numPoints; i++ {
		s.componentOfPoint[i] = s.uf.find(i)
	}

	// Set componentOfNode bottom-up.
	nodeData := s.tree.NodeDataArray()
	idxArray := s.tree.IdxArray()

	for n := s.numNodes - 1; n >= 0; n-- {
		nd := nodeData[n]
		if nd.IsLeaf {
			if nd.IdxStart >= nd.IdxEnd {
				continue
			}
			comp := s.componentOfPoint[idxArray[nd.IdxStart]]
			allSame := true
			for i := nd.IdxStart + 1; i < nd.IdxEnd; i++ {
				if s.componentOfPoint[idxArray[i]] != comp {
					allSame = false
					break
				}
			}
			if allSame {
				s.componentOfNode[n] = comp
			}
		} else {
			left, right := s.tree.ChildNodes(n)
			if s.componentOfNode[left] == s.componentOfNode[right] && s.componentOfNode[left] >= 0 {
				s.componentOfNode[n] = s.componentOfNode[left]
			}
		}
	}

	// Always reset bounds (we don't support approx mode).
	s.components = s.uf.components()
	for i := 0; i < s.numNodes; i++ {
		s.bounds[i] = math.MaxFloat64
	}

	return len(s.components)
}

func (s *boruvkaState) dualTreeTraversal(node1, node2 int) {
	// MinRdistDual returns reduced distance; convert to true distance for comparison.
	nodeDist := s.metric.RdistToDist(s.tree.MinRdistDual(node1, node2))

	// Prune: if node distance >= current bound, nothing useful here.
	if nodeDist >= s.bounds[node1] {
		return
	}
	// Prune: if both nodes are in the same component.
	if s.componentOfNode[node1] == s.componentOfNode[node2] && s.componentOfNode[node1] >= 0 {
		return
	}

	nd := s.tree.NodeDataArray()
	node1Info := nd[node1]
	node2Info := nd[node2]

	// Case 1: Both leaves.
	if node1Info.IsLeaf && node2Info.IsLeaf {
		s.processLeafPair(node1, node2)
		return
	}

	// Case 2a: node1 is a leaf, or node2 is larger → descend into node2.
	if node1Info.IsLeaf || (!node2Info.IsLeaf && nodeSize(node2Info) > nodeSize(node1Info)) {
		left, right := s.tree.ChildNodes(node2)
		leftDist := s.metric.RdistToDist(s.tree.MinRdistDual(node1, left))
		rightDist := s.metric.RdistToDist(s.tree.MinRdistDual(node1, right))
		if leftDist < rightDist {
			s.dualTreeTraversal(node1, left)
			s.dualTreeTraversal(node1, right)
		} else {
			s.dualTreeTraversal(node1, right)
			s.dualTreeTraversal(node1, left)
		}
		return
	}

	// Case 2b: node2 is a leaf, or node1 is larger → descend into node1.
	left, right := s.tree.ChildNodes(node1)
	leftDist := s.metric.RdistToDist(s.tree.MinRdistDual(left, node2))
	rightDist := s.metric.RdistToDist(s.tree.MinRdistDual(right, node2))
	if leftDist < rightDist {
		s.dualTreeTraversal(left, node2)
		s.dualTreeTraversal(right, node2)
	} else {
		s.dualTreeTraversal(right, node2)
		s.dualTreeTraversal(left, node2)
	}
}

func nodeSize(nd NodeData) float64 {
	// For KD-tree (radius==0), use point count as size proxy.
	if nd.Radius > 0 {
		return nd.Radius
	}
	return float64(nd.IdxEnd - nd.IdxStart)
}

func (s *boruvkaState) processLeafPair(node1, node2 int) {
	nd := s.tree.NodeDataArray()
	idxArray := s.tree.IdxArray()
	data := s.tree.Data()
	dims := s.numFeatures

	n1 := nd[node1]
	n2 := nd[node2]

	newUpperBound := 0.0
	newLowerBound := math.MaxFloat64

	for i := n1.IdxStart; i < n1.IdxEnd; i++ {
		p := idxArray[i]
		comp1 := s.componentOfPoint[p]

		// Pruning: if core distance of p already exceeds the best candidate
		// for its component, we can't improve.
		if s.coreDistance[p] > s.candidateDist[comp1] {
			continue
		}

		for j := n2.IdxStart; j < n2.IdxEnd; j++ {
			q := idxArray[j]
			comp2 := s.componentOfPoint[q]

			if s.coreDistance[q] > s.candidateDist[comp1] {
				continue
			}

			if comp1 == comp2 {
				continue
			}

			// Compute true distance between p and q.
			pSlice := data[p*dims : (p+1)*dims]
			qSlice := data[q*dims : (q+1)*dims]
			d := s.metric.Distance(pSlice, qSlice)

			// Mutual reachability distance in true distance space.
			mrDist := d
			if s.alpha != 1.0 {
				mrDist = d / s.alpha
			}
			if s.coreDistance[p] > mrDist {
				mrDist = s.coreDistance[p]
			}
			if s.coreDistance[q] > mrDist {
				mrDist = s.coreDistance[q]
			}

			if mrDist < s.candidateDist[comp1] {
				s.candidateDist[comp1] = mrDist
				s.candidateNeighbor[comp1] = q
				s.candidatePoint[comp1] = p
			}
		}

		if s.candidateDist[comp1] > newUpperBound {
			newUpperBound = s.candidateDist[comp1]
		}
		if s.candidateDist[comp1] < newLowerBound {
			newLowerBound = s.candidateDist[comp1]
		}
	}

	// Compute new bound and propagate up the tree.
	newBound := math.Min(newUpperBound, newLowerBound+2*nd[node1].Radius)

	if newBound < s.bounds[node1] {
		s.bounds[node1] = newBound
		s.propagateBoundsUp(node1)
	}
}

func (s *boruvkaState) propagateBoundsUp(node int) {
	nd := s.tree.NodeDataArray()

	for node > 0 {
		parent := (node - 1) / 2
		left := 2*parent + 1
		right := 2*parent + 2

		parentInfo := nd[parent]
		leftInfo := nd[left]
		rightInfo := nd[right]

		boundMax := math.Max(s.bounds[left], s.bounds[right])
		boundMin := math.Min(
			s.bounds[left]+2*(parentInfo.Radius-leftInfo.Radius),
			s.bounds[right]+2*(parentInfo.Radius-rightInfo.Radius),
		)

		var newBound float64
		if boundMin > 0 {
			newBound = math.Min(boundMax, boundMin)
		} else {
			newBound = boundMax
		}

		if newBound < s.bounds[parent] {
			s.bounds[parent] = newBound
			node = parent
		} else {
			break
		}
	}
}

func (s *boruvkaState) spanningTree() ([][3]float64, []float64) {
	numComponents := len(s.components)
	if numComponents == 0 {
		numComponents = s.numPoints
	}

	for numComponents > 1 && s.numEdges < s.numPoints-1 {
		s.dualTreeTraversal(0, 0)
		numComponents = s.updateComponents()
	}

	// Return a copy of core distances.
	coreDist := make([]float64, s.numPoints)
	copy(coreDist, s.coreDistance)
	return s.edges, coreDist
}

// KDTreeBoruvka performs dual-tree Borůvka MST construction using a KD-tree.
type KDTreeBoruvka struct {
	state *boruvkaState
}

// NewKDTreeBoruvka creates a KDTreeBoruvka instance.
// The provided tree must implement BoruvkaTree.
func NewKDTreeBoruvka(tree BoruvkaTree, metric DistanceMetric, minSamples int, alpha float64) *KDTreeBoruvka {
	return &KDTreeBoruvka{
		state: newBoruvkaState(tree, metric, minSamples, alpha),
	}
}

// SpanningTree computes the MST and returns edges (as [][3]float64 in
// [from, to, weight] format with true distances) and core distances.
func (b *KDTreeBoruvka) SpanningTree() ([][3]float64, []float64) {
	return b.state.spanningTree()
}

// BallTreeBoruvka performs dual-tree Borůvka MST construction using a Ball tree.
type BallTreeBoruvka struct {
	state *boruvkaState
}

// NewBallTreeBoruvka creates a BallTreeBoruvka instance.
func NewBallTreeBoruvka(tree BoruvkaTree, metric DistanceMetric, minSamples int, alpha float64) *BallTreeBoruvka {
	return &BallTreeBoruvka{
		state: newBoruvkaState(tree, metric, minSamples, alpha),
	}
}

// SpanningTree computes the MST and returns edges and core distances.
func (b *BallTreeBoruvka) SpanningTree() ([][3]float64, []float64) {
	return b.state.spanningTree()
}
