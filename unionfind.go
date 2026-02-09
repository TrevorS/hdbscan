package hdbscan

// UnionFind implements a disjoint-set data structure with path compression
// and union by size. It supports 2*n - 1 elements to accommodate
// dendrogram cluster IDs (original points 0..n-1, merged clusters n..2n-2).
type UnionFind struct {
	parent []int
	size   []int
	// nextLabel is the ID for the next merged cluster, starting at n.
	nextLabel int
}

// NewUnionFind creates a UnionFind for n initial elements. The internal
// storage supports up to 2*n - 1 elements to allow the Label function
// to assign new cluster IDs starting at n.
func NewUnionFind(n int) *UnionFind {
	total := 2*n - 1
	if total < 1 {
		total = 1
	}
	parent := make([]int, total)
	size := make([]int, total)
	for i := range parent {
		parent[i] = -1 // -1 means "is a root"
	}
	for i := 0; i < n; i++ {
		size[i] = 1
	}
	return &UnionFind{
		parent:    parent,
		size:      size,
		nextLabel: n,
	}
}

// Find returns the root of the set containing x, with path compression.
func (uf *UnionFind) Find(x int) int {
	// Walk to the root.
	root := x
	for uf.parent[root] != -1 {
		root = uf.parent[root]
	}
	// Path compression: point all nodes along the path directly to root.
	for uf.parent[x] != -1 {
		x, uf.parent[x] = uf.parent[x], root
	}
	return root
}

// Union merges the sets containing x and y by attaching the smaller tree
// under the larger. Returns the new root.
func (uf *UnionFind) Union(x, y int) int {
	rootX := uf.Find(x)
	rootY := uf.Find(y)
	if rootX == rootY {
		return rootX
	}

	// Attach smaller to larger.
	if uf.size[rootX] < uf.size[rootY] {
		rootX, rootY = rootY, rootX
	}
	uf.parent[rootY] = rootX
	uf.size[rootX] += uf.size[rootY]
	return rootX
}
