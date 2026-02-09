package hdbscan

// NodeData describes a single node in a spatial tree.
type NodeData struct {
	IdxStart, IdxEnd int
	IsLeaf           bool
	Radius           float64 // ball tree radius; 0 for KD-tree
}

// SpatialTree is the read interface for KD-trees and Ball trees,
// used by tree-accelerated core distance computation.
type SpatialTree interface {
	// QueryKNN finds the k nearest neighbors for each row in queryData.
	// queryData is flat row-major with queryRows rows.
	// Returns per-query neighbor indices and distances (both sorted by distance).
	QueryKNN(queryData []float64, queryRows, k int) (indices [][]int, distances [][]float64)

	// Data returns the flat row-major point data owned by the tree.
	Data() []float64

	// NumPoints returns the number of points in the tree.
	NumPoints() int

	// NumFeatures returns the dimensionality of each point.
	NumFeatures() int

	// IdxArray returns the permutation array mapping tree-order positions
	// back to original point indices.
	IdxArray() []int

	// NodeDataArray returns the metadata for every node in the tree.
	NodeDataArray() []NodeData
}

// BoruvkaTree extends SpatialTree with operations needed by dual-tree
// Borůvka MST construction.
type BoruvkaTree interface {
	SpatialTree

	// MinRdistDual returns a lower bound (in reduced-distance space) on the
	// distance between any point in node1 and any point in node2.
	MinRdistDual(node1, node2 int) float64

	// MinRdistPoint returns a lower bound (in reduced-distance space) on the
	// distance between a point and any point in the given node.
	MinRdistPoint(node int, point []float64) float64

	// NumNodes returns the total number of nodes (internal + leaf) in the tree.
	NumNodes() int

	// ChildNodes returns the left and right child node indices.
	// Behavior is undefined for leaf nodes.
	ChildNodes(node int) (left, right int)
}
