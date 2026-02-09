package hdbscan

import (
	"math"
	"sort"
)

// SimplifyHierarchy removes low-persistence leaf clusters from the condensed tree.
// persistenceThreshold is the minimum persistence to keep a cluster.
// Returns a new condensed tree with simplified hierarchy.
func SimplifyHierarchy(tree []CondensedTreeEntry, persistenceThreshold float64) []CondensedTreeEntry {
	if persistenceThreshold <= 0 || len(tree) == 0 {
		return copyTree(tree)
	}

	// Find the root cluster (smallest parent ID) and extract cluster entries.
	rootCluster := math.MaxInt
	var clusterEntries []CondensedTreeEntry
	for _, e := range tree {
		if e.Parent < rootCluster {
			rootCluster = e.Parent
		}
		if e.ChildSize > 1 {
			clusterEntries = append(clusterEntries, e)
		}
	}

	if len(clusterEntries) == 0 {
		return copyTree(tree)
	}

	// Build cluster metadata: lambdaBirth and child-to-parent mapping.
	lambdaBirth := make(map[int]float64)
	clusterParent := make(map[int]int)
	for _, e := range clusterEntries {
		lambdaBirth[e.Child] = e.LambdaVal
		clusterParent[e.Child] = e.Parent
	}
	lambdaBirth[rootCluster] = 0.0

	// Iteratively remove low-persistence leaves until stable.
	removed := make(map[int]bool)
	changed := true
	for changed {
		changed = false

		// Identify current leaf clusters (children that are not parents of
		// any surviving cluster entry).
		survivingParents := make(map[int]bool)
		for _, e := range clusterEntries {
			if !removed[e.Parent] && !removed[e.Child] {
				survivingParents[e.Parent] = true
			}
		}

		for _, e := range clusterEntries {
			child := e.Child
			if removed[child] || survivingParents[child] {
				continue
			}
			persistence := lambdaBirth[child] - lambdaBirth[clusterParent[child]]
			if persistence < persistenceThreshold {
				removed[child] = true
				changed = true
			}
		}
	}

	if len(removed) == 0 {
		return copyTree(tree)
	}

	// For each removed cluster, find the nearest surviving ancestor.
	reparent := make(map[int]int)
	for c := range removed {
		ancestor := clusterParent[c]
		for removed[ancestor] {
			ancestor = clusterParent[ancestor]
		}
		reparent[c] = ancestor
	}

	// Build the new tree: drop removed cluster entries, re-parent survivors.
	var newTree []CondensedTreeEntry
	for _, e := range tree {
		if e.ChildSize > 1 && removed[e.Child] {
			continue
		}
		parent := e.Parent
		if removed[parent] {
			parent = reparent[parent]
		}
		newTree = append(newTree, CondensedTreeEntry{
			Parent:    parent,
			Child:     e.Child,
			LambdaVal: e.LambdaVal,
			ChildSize: e.ChildSize,
		})
	}

	// Relabel clusters for consecutive numbering starting at rootCluster.
	relabelClusters(newTree, rootCluster)

	return newTree
}

// copyTree returns a shallow copy of tree.
func copyTree(tree []CondensedTreeEntry) []CondensedTreeEntry {
	result := make([]CondensedTreeEntry, len(tree))
	copy(result, tree)
	return result
}

// relabelClusters renumbers cluster IDs in-place so they are consecutive
// starting at startID, preserving their relative order.
func relabelClusters(tree []CondensedTreeEntry, startID int) {
	clusterIDs := make(map[int]bool)
	for _, e := range tree {
		clusterIDs[e.Parent] = true
		if e.ChildSize > 1 {
			clusterIDs[e.Child] = true
		}
	}

	sorted := make([]int, 0, len(clusterIDs))
	for c := range clusterIDs {
		sorted = append(sorted, c)
	}
	sort.Ints(sorted)

	relabel := make(map[int]int, len(sorted))
	for i, c := range sorted {
		relabel[c] = startID + i
	}

	for i := range tree {
		tree[i].Parent = relabel[tree[i].Parent]
		if tree[i].ChildSize > 1 {
			tree[i].Child = relabel[tree[i].Child]
		}
	}
}
