package hdbscan

import "math"

// CondensedTreeEntry represents a single entry in the condensed tree.
type CondensedTreeEntry struct {
	Parent    int
	Child     int
	LambdaVal float64
	ChildSize int
}

// CondenseTree converts a single-linkage dendrogram into a condensed tree
// by collapsing clusters smaller than minClusterSize.
// dendrogram is [][4]float64 in scipy format: [left, right, distance, mergedSize].
func CondenseTree(dendrogram [][4]float64, minClusterSize int) []CondensedTreeEntry {
	numRows := len(dendrogram)
	if numRows == 0 {
		return nil
	}

	numPoints := numRows + 1
	root := 2 * numRows
	nextLabel := numPoints + 1

	nodeList := bfsFromHierarchy(dendrogram, root, numPoints)

	relabel := make(map[int]int)
	relabel[root] = numPoints

	ignore := make(map[int]bool)

	var result []CondensedTreeEntry

	// collapseSubtree walks the subtree rooted at node, emitting a point entry
	// for each leaf and marking every visited node as ignored.
	collapseSubtree := func(subtreeRoot, parentCluster int, lambda float64) {
		for _, subNode := range bfsFromHierarchy(dendrogram, subtreeRoot, numPoints) {
			if subNode < numPoints {
				result = append(result, CondensedTreeEntry{
					Parent:    parentCluster,
					Child:     subNode,
					LambdaVal: lambda,
					ChildSize: 1,
				})
			}
			ignore[subNode] = true
		}
	}

	for _, node := range nodeList {
		if ignore[node] || node < numPoints {
			continue
		}

		row := dendrogram[node-numPoints]
		left := int(row[0])
		right := int(row[1])
		dist := row[2]

		var lambda float64
		if dist > 0.0 {
			lambda = 1.0 / dist
		} else {
			lambda = math.Inf(1)
		}

		leftCount := 1
		if left >= numPoints {
			leftCount = int(dendrogram[left-numPoints][3])
		}

		rightCount := 1
		if right >= numPoints {
			rightCount = int(dendrogram[right-numPoints][3])
		}

		leftBig := leftCount >= minClusterSize
		rightBig := rightCount >= minClusterSize
		parentCluster := relabel[node]

		switch {
		case leftBig && rightBig:
			relabel[left] = nextLabel
			nextLabel++
			result = append(result, CondensedTreeEntry{
				Parent:    parentCluster,
				Child:     relabel[left],
				LambdaVal: lambda,
				ChildSize: leftCount,
			})

			relabel[right] = nextLabel
			nextLabel++
			result = append(result, CondensedTreeEntry{
				Parent:    parentCluster,
				Child:     relabel[right],
				LambdaVal: lambda,
				ChildSize: rightCount,
			})

		case !leftBig && !rightBig:
			collapseSubtree(left, parentCluster, lambda)
			collapseSubtree(right, parentCluster, lambda)

		case !leftBig:
			// Left too small; right continues as same cluster
			relabel[right] = parentCluster
			collapseSubtree(left, parentCluster, lambda)

		default:
			// Right too small; left continues as same cluster
			relabel[left] = parentCluster
			collapseSubtree(right, parentCluster, lambda)
		}
	}

	return result
}

// bfsFromHierarchy performs a breadth-first search on a scipy-format hierarchy,
// returning all node IDs reachable from bfsRoot.
func bfsFromHierarchy(hierarchy [][4]float64, bfsRoot, numPoints int) []int {
	toProcess := []int{bfsRoot}
	var result []int

	for len(toProcess) > 0 {
		result = append(result, toProcess...)

		// Get dendrogram row indices for internal nodes
		var nextLevel []int
		for _, x := range toProcess {
			if x >= numPoints {
				idx := x - numPoints
				if idx < len(hierarchy) {
					nextLevel = append(nextLevel, idx)
				}
			}
		}

		if len(nextLevel) == 0 {
			break
		}

		toProcess = toProcess[:0]
		for _, idx := range nextLevel {
			row := hierarchy[idx]
			toProcess = append(toProcess, int(row[0]), int(row[1]))
		}
	}

	return result
}
