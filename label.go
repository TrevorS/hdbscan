package hdbscan

import "sort"

// Label converts sorted MST edges into a single-linkage dendrogram in scipy
// format. mstEdges is [][3]float64 where each edge is [from, to, weight].
// Returns [][4]float64 dendrogram rows: [left, right, distance, mergedSize].
// New cluster IDs start at n and increment. The dendrogram uses the same
// cluster-ID scheme as scipy's linkage output.
func Label(mstEdges [][3]float64, n int) [][4]float64 {
	if len(mstEdges) == 0 {
		return nil
	}

	// Sort edges by weight ascending.
	sorted := make([][3]float64, len(mstEdges))
	copy(sorted, mstEdges)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i][2] < sorted[j][2]
	})

	// Use a UnionFind with 2*n - 1 elements so that merged cluster IDs
	// (n, n+1, ...) can be stored as union-find roots.
	uf := NewUnionFind(n)

	result := make([][4]float64, 0, len(sorted))

	for _, edge := range sorted {
		a := int(edge[0])
		b := int(edge[1])
		weight := edge[2]

		aa := uf.Find(a)
		bb := uf.Find(b)
		newSize := uf.size[aa] + uf.size[bb]

		result = append(result, [4]float64{float64(aa), float64(bb), weight, float64(newSize)})

		// Relabel the merged root to the next dendrogram cluster ID
		// (n + index). Both parents point to nextLabel, matching the
		// reference implementation's mst_linkage_core_vector output.
		uf.size[uf.nextLabel] = newSize
		uf.parent[aa] = uf.nextLabel
		uf.parent[bb] = uf.nextLabel
		uf.nextLabel++
	}

	return result
}
