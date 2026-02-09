package hdbscan

import (
	"math"
	"testing"
)

// buildMSTVectorFromPoints is a helper that computes core distances from a full
// distance matrix, then runs PrimMSTVector on the raw point data.
func buildMSTVectorFromPoints(data []float64, n, dims, minSamples int, metric DistanceMetric, alpha float64) [][3]float64 {
	distMatrix := ComputePairwiseDistances(data, n, dims, metric)
	coreDistances := ComputeCoreDistances(distMatrix, n, minSamples)
	return PrimMSTVector(data, n, dims, coreDistances, metric, alpha)
}

// buildMSTBruteFromPoints is a helper that computes the brute-force MST via the
// full distance matrix path for comparison.
func buildMSTBruteFromPoints(data []float64, n, dims, minSamples int, metric DistanceMetric, alpha float64) [][3]float64 {
	distMatrix := ComputePairwiseDistances(data, n, dims, metric)
	coreDistances := ComputeCoreDistances(distMatrix, n, minSamples)
	mrMatrix := MutualReachability(distMatrix, coreDistances, n, alpha)
	return PrimMST(mrMatrix, n)
}

func TestPrimMSTVector_TwoPoints(t *testing.T) {
	// Two points: (0,0) and (3,4), distance = 5.
	data := []float64{0, 0, 3, 4}
	n, dims := 2, 2

	edges := buildMSTVectorFromPoints(data, n, dims, 1, EuclideanMetric{}, 1.0)

	if len(edges) != 1 {
		t.Fatalf("expected 1 edge for n=2, got %d", len(edges))
	}
	if math.Abs(edges[0][2]-5.0) > 1e-10 {
		t.Errorf("expected edge weight 5.0, got %f", edges[0][2])
	}
}

func TestPrimMSTVector_SinglePoint(t *testing.T) {
	data := []float64{1, 2}
	coreDistances := []float64{0}
	edges := PrimMSTVector(data, 1, 2, coreDistances, EuclideanMetric{}, 1.0)
	if len(edges) != 0 {
		t.Fatalf("expected 0 edges for n=1, got %d", len(edges))
	}
}

func TestPrimMSTVector_MatchesBruteForceWeight(t *testing.T) {
	// 6-point dataset. PrimMSTVector should produce an MST with the same
	// total weight as the brute-force PrimMST path.
	data := []float64{
		0, 0,
		1, 0,
		5, 0,
		6, 0,
		3, 3,
		3, -3,
	}
	n, dims := 6, 2

	for _, minSamples := range []int{1, 2, 3, 5} {
		vectorEdges := buildMSTVectorFromPoints(data, n, dims, minSamples, EuclideanMetric{}, 1.0)
		bruteEdges := buildMSTBruteFromPoints(data, n, dims, minSamples, EuclideanMetric{}, 1.0)

		vectorWeight := totalMSTWeight(vectorEdges)
		bruteWeight := totalMSTWeight(bruteEdges)

		if math.Abs(vectorWeight-bruteWeight) > 1e-10 {
			t.Errorf("minSamples=%d: vector MST weight %f != brute MST weight %f",
				minSamples, vectorWeight, bruteWeight)
		}
	}
}

func TestPrimMSTVector_AlphaNotOne(t *testing.T) {
	// Verify that alpha != 1.0 produces the same MST weight as brute-force.
	data := []float64{
		0, 0,
		1, 0,
		5, 0,
		6, 0,
		3, 3,
		3, -3,
	}
	n, dims := 6, 2
	alpha := 0.5

	vectorEdges := buildMSTVectorFromPoints(data, n, dims, 2, EuclideanMetric{}, alpha)
	bruteEdges := buildMSTBruteFromPoints(data, n, dims, 2, EuclideanMetric{}, alpha)

	vectorWeight := totalMSTWeight(vectorEdges)
	bruteWeight := totalMSTWeight(bruteEdges)

	if math.Abs(vectorWeight-bruteWeight) > 1e-10 {
		t.Errorf("alpha=%f: vector MST weight %f != brute MST weight %f",
			alpha, vectorWeight, bruteWeight)
	}
}

func TestPrimMSTVector_IdenticalPoints(t *testing.T) {
	// All points are the same. Core distances are 0, so mutual reachability
	// distances are all 0.
	data := []float64{
		1, 2,
		1, 2,
		1, 2,
		1, 2,
	}
	n, dims := 4, 2

	vectorEdges := buildMSTVectorFromPoints(data, n, dims, 1, EuclideanMetric{}, 1.0)

	if len(vectorEdges) != 3 {
		t.Fatalf("expected 3 edges, got %d", len(vectorEdges))
	}
	for i, e := range vectorEdges {
		if e[2] != 0 {
			t.Errorf("edge %d: expected weight 0, got %f", i, e[2])
		}
	}
}

func TestPrimMSTVector_ManhattanMetric(t *testing.T) {
	data := []float64{
		0, 0,
		3, 4,
		6, 0,
	}
	n, dims := 3, 2

	vectorEdges := buildMSTVectorFromPoints(data, n, dims, 1, ManhattanMetric{}, 1.0)
	bruteEdges := buildMSTBruteFromPoints(data, n, dims, 1, ManhattanMetric{}, 1.0)

	vectorWeight := totalMSTWeight(vectorEdges)
	bruteWeight := totalMSTWeight(bruteEdges)

	if math.Abs(vectorWeight-bruteWeight) > 1e-10 {
		t.Errorf("Manhattan: vector MST weight %f != brute MST weight %f",
			vectorWeight, bruteWeight)
	}
}

func TestPrimMSTVector_FullPipelineMatch(t *testing.T) {
	// Run the full pipeline through both paths and verify labels match.
	data := []float64{
		0, 0,
		0.1, 0,
		0.2, 0,
		10, 0,
		10.1, 0,
		10.2, 0,
	}
	n, dims := 6, 2
	minSamples := 2
	metric := EuclideanMetric{}
	alpha := 1.0

	// Brute-force path.
	distMatrix := ComputePairwiseDistances(data, n, dims, metric)
	coreDistances := ComputeCoreDistances(distMatrix, n, minSamples)
	mrMatrix := MutualReachability(distMatrix, coreDistances, n, alpha)
	bruteEdges := PrimMST(mrMatrix, n)
	bruteDendrogram := Label(bruteEdges, n)

	// Vector path.
	vectorEdges := PrimMSTVector(data, n, dims, coreDistances, metric, alpha)
	vectorDendrogram := Label(vectorEdges, n)

	if len(bruteDendrogram) != len(vectorDendrogram) {
		t.Fatalf("dendrogram length mismatch: brute=%d vector=%d",
			len(bruteDendrogram), len(vectorDendrogram))
	}

	// Check that dendrogram merge distances are the same (order may differ
	// due to tie-breaking, but total set of merge distances should match).
	bruteDists := make([]float64, len(bruteDendrogram))
	vectorDists := make([]float64, len(vectorDendrogram))
	for i := range bruteDendrogram {
		bruteDists[i] = bruteDendrogram[i][2]
		vectorDists[i] = vectorDendrogram[i][2]
	}

	// Sort and compare.
	sortFloat64s(bruteDists)
	sortFloat64s(vectorDists)

	for i := range bruteDists {
		if math.Abs(bruteDists[i]-vectorDists[i]) > 1e-10 {
			t.Errorf("dendrogram distance mismatch at index %d: brute=%f vector=%f",
				i, bruteDists[i], vectorDists[i])
		}
	}
}

// sortFloat64s sorts a float64 slice in ascending order (inline, no import needed
// beyond what mst_test.go already uses).
func sortFloat64s(s []float64) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

func TestPrimMSTVector_EdgeFormat(t *testing.T) {
	// Verify that the "from" field of each edge is a valid node index
	// (not just the previously-added node).
	data := []float64{
		0, 0,
		1, 0,
		10, 0,
		11, 0,
	}
	n, dims := 4, 2

	edges := buildMSTVectorFromPoints(data, n, dims, 1, EuclideanMetric{}, 1.0)

	for i, e := range edges {
		from := int(e[0])
		to := int(e[1])
		if from < 0 || from >= n {
			t.Errorf("edge %d: invalid from=%d", i, from)
		}
		if to < 0 || to >= n {
			t.Errorf("edge %d: invalid to=%d", i, to)
		}
		if from == to {
			t.Errorf("edge %d: self-loop from=%d to=%d", i, from, to)
		}
	}
}
