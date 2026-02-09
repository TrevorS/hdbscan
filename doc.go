// Package hdbscan implements Hierarchical Density-Based Spatial Clustering
// of Applications with Noise (HDBSCAN).
//
// HDBSCAN extends DBSCAN by converting it into a hierarchical algorithm and
// then extracting a flat clustering based on cluster stability. It can find
// clusters of varying densities and robustly identifies noise points.
//
// Basic usage:
//
//	cfg := hdbscan.DefaultConfig()
//	cfg.MinClusterSize = 10
//	result, err := hdbscan.Cluster(data, cfg)
//	// result.Labels[i] is the cluster ID for point i (-1 = noise)
//	// result.Probabilities[i] is how strongly point i belongs to its cluster
//	// result.OutlierScores[i] is how outlier-like point i is (0 = inlier, 1 = outlier)
//
// For precomputed distance matrices:
//
//	result, err := hdbscan.ClusterPrecomputed(distMatrix, n, cfg)
//
// # Algorithm selection
//
// By default (Algorithm: "auto"), Cluster picks the best MST construction
// strategy based on the metric and dimensionality. For standard metrics on
// low-dimensional data, it uses dual-tree Borůvka with a KD-tree, which is
// significantly faster and uses O(n) memory instead of O(n²). Set
// Config.Algorithm to force a specific strategy:
//
//	cfg.Algorithm = hdbscan.AlgorithmBrute          // full distance matrix
//	cfg.Algorithm = hdbscan.AlgorithmBoruvkaKDTree  // dual-tree Borůvka + KD-tree
//	cfg.Algorithm = hdbscan.AlgorithmPrimsKDTree    // matrix-free Prim's + KD-tree
package hdbscan
