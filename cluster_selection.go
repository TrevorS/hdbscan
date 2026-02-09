package hdbscan

import (
	"math"
	"sort"
)

// treeRoot returns the root cluster ID (smallest parent) from the condensed tree.
func treeRoot(tree []CondensedTreeEntry) int {
	root := math.MaxInt
	for _, e := range tree {
		if e.Parent < root {
			root = e.Parent
		}
	}
	return root
}

// clusterEntries returns only the cluster-to-cluster entries (ChildSize > 1).
func clusterEntries(tree []CondensedTreeEntry) []CondensedTreeEntry {
	entries := make([]CondensedTreeEntry, 0, len(tree)/2)
	for _, e := range tree {
		if e.ChildSize > 1 {
			entries = append(entries, e)
		}
	}
	return entries
}

// clusterChildrenMap builds a parent-to-children mapping from cluster entries.
func clusterChildrenMap(clusterTree []CondensedTreeEntry) map[int][]int {
	childrenOf := make(map[int][]int)
	for _, e := range clusterTree {
		childrenOf[e.Parent] = append(childrenOf[e.Parent], e.Child)
	}
	return childrenOf
}

// bfsDescendants returns all cluster descendants of a node (including itself)
// using a pre-built children map.
func bfsDescendants(childrenOf map[int][]int, bfsRoot int) []int {
	result := []int{bfsRoot}
	toProcess := []int{bfsRoot}

	for len(toProcess) > 0 {
		var next []int
		for _, node := range toProcess {
			for _, child := range childrenOf[node] {
				result = append(result, child)
				next = append(next, child)
			}
		}
		toProcess = next
	}

	return result
}

// SelectClustersEOM performs Excess-of-Mass cluster selection.
// It walks the condensed tree bottom-up, selecting clusters that maximize total stability.
//
// Parameters:
//   - tree: the condensed tree
//   - stability: stability map from ComputeStability
//   - allowSingleCluster: if true, the root cluster is included as a candidate
//   - maxClusterSize: if > 0, clusters larger than this force children to win
//   - clusterSelectionEpsilonMax: if a cluster's epsilon (1/lambdaBirth) exceeds this, children win
//
// Returns selected cluster IDs and updated stability map.
func SelectClustersEOM(tree []CondensedTreeEntry, stability map[int]float64,
	allowSingleCluster bool, maxClusterSize int, clusterSelectionEpsilonMax float64,
) (map[int]bool, map[int]float64) {
	stab := make(map[int]float64, len(stability))
	for k, v := range stability {
		stab[k] = v
	}

	root := treeRoot(tree)
	clusterTree := clusterEntries(tree)
	childrenOf := clusterChildrenMap(clusterTree)

	// Cluster sizes from cluster entries.
	clusterSizes := make(map[int]int)
	for _, e := range clusterTree {
		clusterSizes[e.Child] = e.ChildSize
	}
	if allowSingleCluster {
		rootSize := 0
		for _, e := range clusterTree {
			if e.Parent == root {
				rootSize += e.ChildSize
			}
		}
		clusterSizes[root] = rootSize
	}

	// Cluster epsilon (1/lambdaBirth).
	nodeEps := make(map[int]float64)
	for _, e := range clusterTree {
		nodeEps[e.Child] = 1.0 / e.LambdaVal
	}
	if allowSingleCluster {
		maxInvLambda := 0.0
		for _, e := range tree {
			if e.LambdaVal > 0 {
				if invL := 1.0 / e.LambdaVal; invL > maxInvLambda {
					maxInvLambda = invL
				}
			}
		}
		nodeEps[root] = maxInvLambda
	}

	// Build candidate list sorted in reverse topological order.
	// Cluster IDs are assigned in BFS order so reverse numeric sort works.
	var nodeList []int
	for k := range stab {
		if allowSingleCluster || k != root {
			nodeList = append(nodeList, k)
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(nodeList)))

	isCluster := make(map[int]bool, len(nodeList))
	for _, n := range nodeList {
		isCluster[n] = true
	}

	if maxClusterSize <= 0 {
		maxPoints := 0
		for _, e := range tree {
			if e.ChildSize == 1 && e.Child+1 > maxPoints {
				maxPoints = e.Child + 1
			}
		}
		maxClusterSize = maxPoints + 1
	}

	// EOM selection: bottom-up.
	for _, node := range nodeList {
		children := childrenOf[node]
		if len(children) == 0 {
			continue
		}

		subtreeStability := 0.0
		for _, child := range children {
			subtreeStability += stab[child]
		}

		childrenWin := subtreeStability > stab[node] ||
			clusterSizes[node] > maxClusterSize ||
			nodeEps[node] > clusterSelectionEpsilonMax

		if childrenWin {
			isCluster[node] = false
			stab[node] = subtreeStability
		} else {
			// Parent wins -- deselect all descendants.
			for _, d := range bfsDescendants(childrenOf, node) {
				if d != node {
					isCluster[d] = false
				}
			}
		}
	}

	selected := make(map[int]bool)
	for k, v := range isCluster {
		if v {
			selected[k] = true
		}
	}

	return selected, stab
}

// SelectClustersLeaf selects all leaf clusters from the condensed tree.
// If clusterSelectionEpsilon > 0, applies epsilon search to merge small-epsilon leaves.
func SelectClustersLeaf(tree []CondensedTreeEntry, clusterSelectionEpsilon float64) map[int]bool {
	clusterTree := clusterEntries(tree)
	leaves := getClusterTreeLeaves(clusterTree)

	if len(leaves) == 0 {
		return map[int]bool{treeRoot(tree): true}
	}

	if clusterSelectionEpsilon > 0 {
		return EpsilonSearch(tree, leaves, clusterSelectionEpsilon, false)
	}

	return leaves
}

// EpsilonSearch adjusts selected clusters based on epsilon threshold.
// For candidates with epsilon (1/lambdaBirth) below threshold, it traverses upward
// to find an ancestor whose epsilon meets the threshold.
func EpsilonSearch(tree []CondensedTreeEntry, candidateClusters map[int]bool,
	clusterSelectionEpsilon float64, allowSingleCluster bool,
) map[int]bool {
	clusterTree := clusterEntries(tree)
	root := treeRoot(tree)
	childrenOf := clusterChildrenMap(clusterTree)

	// child-to-parent and child-to-lambda mappings for upward traversal.
	childToParent := make(map[int]int)
	childToLambda := make(map[int]float64)
	for _, e := range clusterTree {
		childToParent[e.Child] = e.Parent
		childToLambda[e.Child] = e.LambdaVal
	}

	processed := make(map[int]bool)
	result := make(map[int]bool)

	for leaf := range candidateClusters {
		lambda, ok := childToLambda[leaf]
		if !ok {
			result[leaf] = true
			continue
		}

		eps := 1.0 / lambda
		if eps >= clusterSelectionEpsilon {
			result[leaf] = true
			continue
		}

		if processed[leaf] {
			continue
		}

		epsilonChild := traverseUpwards(childToParent, childToLambda, root, clusterSelectionEpsilon, leaf, allowSingleCluster)
		result[epsilonChild] = true

		// Mark all descendants of epsilonChild as processed.
		for _, subNode := range bfsDescendants(childrenOf, epsilonChild) {
			if subNode != epsilonChild {
				processed[subNode] = true
			}
		}
	}

	return result
}

// traverseUpwards walks from a leaf cluster up to find an ancestor whose epsilon >= threshold.
func traverseUpwards(childToParent map[int]int, childToLambda map[int]float64,
	root int, clusterSelectionEpsilon float64, leaf int, allowSingleCluster bool,
) int {
	parent, ok := childToParent[leaf]
	if !ok {
		return leaf
	}
	if parent == root {
		if allowSingleCluster {
			return parent
		}
		return leaf
	}

	parentLambda, ok := childToLambda[parent]
	if !ok {
		return leaf
	}
	if 1.0/parentLambda > clusterSelectionEpsilon {
		return parent
	}
	return traverseUpwards(childToParent, childToLambda, root, clusterSelectionEpsilon, parent, allowSingleCluster)
}

// getClusterTreeLeaves returns leaf cluster IDs from a cluster-only tree.
func getClusterTreeLeaves(clusterTree []CondensedTreeEntry) map[int]bool {
	if len(clusterTree) == 0 {
		return nil
	}

	isParent := make(map[int]bool)
	for _, e := range clusterTree {
		isParent[e.Parent] = true
	}

	leaves := make(map[int]bool)
	for _, e := range clusterTree {
		if !isParent[e.Child] {
			leaves[e.Child] = true
		}
	}

	return leaves
}
