package hdbscan

import (
	"math"
	"sort"
)

// GetLabelsAndProbabilities assigns cluster labels and membership probabilities
// to each data point given the selected clusters from the condensed tree.
//
// Parameters:
//   - tree: the condensed tree
//   - selectedClusters: set of selected cluster IDs
//   - n: number of data points
//   - allowSingleCluster: if true, a single root cluster is valid
//   - clusterSelectionEpsilon: epsilon threshold (0 means unused)
//   - matchReferenceImplementation: match edge-case behavior of reference
//
// Returns labels (noise = -1) and probabilities in [0, 1].
func GetLabelsAndProbabilities(tree []CondensedTreeEntry, selectedClusters map[int]bool,
	n int, allowSingleCluster bool, clusterSelectionEpsilon float64,
	matchReferenceImplementation bool,
) ([]int, []float64) {
	if len(tree) == 0 {
		labels := make([]int, n)
		probs := make([]float64, n)
		for i := range labels {
			labels[i] = -1
		}
		return labels, probs
	}

	rootCluster := treeRoot(tree)
	deaths := computeMaxLambdas(tree)

	// Build cluster label map: sorted selected cluster IDs to sequential labels.
	clusterLabelMap, reverseClusterMap := buildClusterLabelMaps(selectedClusters)

	labels := doLabelling(tree, selectedClusters, clusterLabelMap, rootCluster, n,
		allowSingleCluster, clusterSelectionEpsilon, matchReferenceImplementation)

	probs := getProbabilities(tree, reverseClusterMap, labels, deaths, rootCluster)

	return labels, probs
}

// buildClusterLabelMaps builds bidirectional maps between cluster IDs and
// sequential label indices (0, 1, 2, ...).
func buildClusterLabelMaps(clusters map[int]bool) (forward map[int]int, reverse map[int]int) {
	sorted := make([]int, 0, len(clusters))
	for c := range clusters {
		sorted = append(sorted, c)
	}
	sort.Ints(sorted)

	forward = make(map[int]int, len(sorted))
	reverse = make(map[int]int, len(sorted))
	for i, c := range sorted {
		forward[c] = i
		reverse[i] = c
	}
	return forward, reverse
}

// doLabelling assigns each point to a cluster using union-find on the condensed tree.
func doLabelling(tree []CondensedTreeEntry, clusters map[int]bool,
	clusterLabelMap map[int]int, rootCluster, n int, allowSingleCluster bool,
	clusterSelectionEpsilon float64, matchReferenceImplementation bool,
) []int {
	maxNode := 0
	for _, e := range tree {
		if e.Parent > maxNode {
			maxNode = e.Parent
		}
		if e.Child > maxNode {
			maxNode = e.Child
		}
	}

	uf := newLabelUnionFind(maxNode + 1)
	for _, e := range tree {
		if !clusters[e.Child] {
			uf.union(e.Parent, e.Child)
		}
	}

	// Point lambdas and cluster birth lambdas for edge-case handling.
	pointLambdas := make(map[int]float64)
	clusterBirthLambdas := make(map[int]float64)
	for _, e := range tree {
		if e.ChildSize == 1 {
			pointLambdas[e.Child] = e.LambdaVal
		} else {
			clusterBirthLambdas[e.Child] = e.LambdaVal
		}
	}

	// Max lambda among points directly under the root cluster.
	rootMaxPointLambda := 0.0
	for _, e := range tree {
		if e.Parent == rootCluster && e.ChildSize == 1 && e.LambdaVal > rootMaxPointLambda {
			rootMaxPointLambda = e.LambdaVal
		}
	}

	result := make([]int, n)
	for i := 0; i < n; i++ {
		result[i] = labelPoint(i, uf.find(i), rootCluster, clusters, clusterLabelMap,
			pointLambdas, clusterBirthLambdas, rootMaxPointLambda,
			allowSingleCluster, clusterSelectionEpsilon, matchReferenceImplementation)
	}

	return result
}

// labelPoint determines the label for a single point based on its union-find cluster.
func labelPoint(point, cluster, rootCluster int,
	clusters map[int]bool, clusterLabelMap map[int]int,
	pointLambdas map[int]float64, clusterBirthLambdas map[int]float64,
	rootMaxPointLambda float64,
	allowSingleCluster bool, clusterSelectionEpsilon float64,
	matchReferenceImplementation bool,
) int {
	if cluster < rootCluster {
		return -1
	}

	if cluster != rootCluster {
		if !matchReferenceImplementation {
			return clusterLabelMap[cluster]
		}
		if pointLambdas[point] > clusterBirthLambdas[cluster] {
			return clusterLabelMap[cluster]
		}
		return -1
	}

	// Point resolved to root cluster.
	if len(clusters) != 1 || !allowSingleCluster {
		return -1
	}
	label, ok := clusterLabelMap[cluster]
	if !ok {
		return -1
	}

	pointLambda := pointLambdas[point]
	if clusterSelectionEpsilon != 0.0 {
		if pointLambda >= 1.0/clusterSelectionEpsilon {
			return label
		}
		return -1
	}
	if pointLambda >= rootMaxPointLambda {
		return label
	}
	return -1
}

// getProbabilities computes membership probability for each point.
func getProbabilities(tree []CondensedTreeEntry, reverseClusterMap map[int]int,
	labels []int, deaths map[int]float64, rootCluster int,
) []float64 {
	result := make([]float64, len(labels))

	for _, e := range tree {
		point := e.Child
		if point >= rootCluster {
			continue
		}

		clusterNum := labels[point]
		if clusterNum == -1 {
			continue
		}

		cluster, ok := reverseClusterMap[clusterNum]
		if !ok {
			continue
		}

		maxLambda := deaths[cluster]
		if maxLambda == 0.0 || math.IsInf(e.LambdaVal, 0) {
			result[point] = 1.0
		} else {
			result[point] = math.Min(e.LambdaVal, maxLambda) / maxLambda
		}
	}

	return result
}

// computeMaxLambdas computes the max lambda (death) for each cluster.
func computeMaxLambdas(tree []CondensedTreeEntry) map[int]float64 {
	deaths := make(map[int]float64)
	for _, e := range tree {
		if e.LambdaVal > deaths[e.Parent] {
			deaths[e.Parent] = e.LambdaVal
		}
	}
	return deaths
}

// labelUnionFind is a simple union-find for labelling.
type labelUnionFind struct {
	parent []int
	rank   []int
}

func newLabelUnionFind(size int) *labelUnionFind {
	parent := make([]int, size)
	rank := make([]int, size)
	for i := range parent {
		parent[i] = i
	}
	return &labelUnionFind{parent: parent, rank: rank}
}

func (uf *labelUnionFind) find(x int) int {
	if uf.parent[x] != x {
		uf.parent[x] = uf.find(uf.parent[x])
	}
	return uf.parent[x]
}

func (uf *labelUnionFind) union(x, y int) {
	xRoot := uf.find(x)
	yRoot := uf.find(y)
	if xRoot == yRoot {
		return
	}
	if uf.rank[xRoot] < uf.rank[yRoot] {
		uf.parent[xRoot] = yRoot
	} else if uf.rank[xRoot] > uf.rank[yRoot] {
		uf.parent[yRoot] = xRoot
	} else {
		uf.parent[yRoot] = xRoot
		uf.rank[xRoot]++
	}
}
