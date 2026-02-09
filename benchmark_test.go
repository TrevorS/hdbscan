package hdbscan

import (
	"math/rand"
	"testing"
)

func generateBenchData(n, dims int) [][]float64 {
	rng := rand.New(rand.NewSource(42))
	data := make([][]float64, n)
	for i := range data {
		data[i] = make([]float64, dims)
		for j := range data[i] {
			data[i][j] = rng.Float64() * 100
		}
	}
	return data
}

func generateFlatData(n, dims int) []float64 {
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, n*dims)
	for i := range data {
		data[i] = rng.Float64() * 100
	}
	return data
}

// --- Pairwise Distances ---

func benchPairwiseDistances(b *testing.B, n int) {
	b.Helper()
	dims := 2
	data := generateFlatData(n, dims)
	metric := EuclideanMetric{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputePairwiseDistances(data, n, dims, metric)
	}
}

func BenchmarkPairwiseDistances_100(b *testing.B)  { benchPairwiseDistances(b, 100) }
func BenchmarkPairwiseDistances_500(b *testing.B)  { benchPairwiseDistances(b, 500) }
func BenchmarkPairwiseDistances_1000(b *testing.B) { benchPairwiseDistances(b, 1000) }

// --- Core Distances ---

func benchCoreDistances(b *testing.B, n int) {
	b.Helper()
	dims := 2
	data := generateFlatData(n, dims)
	metric := EuclideanMetric{}
	distMatrix := ComputePairwiseDistances(data, n, dims, metric)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeCoreDistances(distMatrix, n, 5)
	}
}

func BenchmarkCoreDistances_100(b *testing.B) { benchCoreDistances(b, 100) }
func BenchmarkCoreDistances_500(b *testing.B) { benchCoreDistances(b, 500) }

// --- Mutual Reachability ---

func benchMutualReachability(b *testing.B, n int) {
	b.Helper()
	dims := 2
	data := generateFlatData(n, dims)
	metric := EuclideanMetric{}
	distMatrix := ComputePairwiseDistances(data, n, dims, metric)
	coreDistances := ComputeCoreDistances(distMatrix, n, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MutualReachability(distMatrix, coreDistances, n, 1.0)
	}
}

func BenchmarkMutualReachability_100(b *testing.B) { benchMutualReachability(b, 100) }

// --- Prim's MST ---

func benchPrimMST(b *testing.B, n int) {
	b.Helper()
	dims := 2
	data := generateFlatData(n, dims)
	metric := EuclideanMetric{}
	distMatrix := ComputePairwiseDistances(data, n, dims, metric)
	coreDistances := ComputeCoreDistances(distMatrix, n, 5)
	mrMatrix := MutualReachability(distMatrix, coreDistances, n, 1.0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PrimMST(mrMatrix, n)
	}
}

func BenchmarkPrimMST_100(b *testing.B) { benchPrimMST(b, 100) }
func BenchmarkPrimMST_500(b *testing.B) { benchPrimMST(b, 500) }

// --- Condense Tree ---

func benchCondenseTree(b *testing.B, n int) {
	b.Helper()
	dims := 2
	data := generateFlatData(n, dims)
	metric := EuclideanMetric{}
	distMatrix := ComputePairwiseDistances(data, n, dims, metric)
	coreDistances := ComputeCoreDistances(distMatrix, n, 5)
	mrMatrix := MutualReachability(distMatrix, coreDistances, n, 1.0)
	mstEdges := PrimMST(mrMatrix, n)
	dendrogram := Label(mstEdges, n)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CondenseTree(dendrogram, 5)
	}
}

func BenchmarkCondenseTree_100(b *testing.B) { benchCondenseTree(b, 100) }

// --- Full Pipeline ---

func benchFullPipeline(b *testing.B, n int) {
	b.Helper()
	dims := 2
	data := generateBenchData(n, dims)
	cfg := DefaultConfig()
	cfg.MinClusterSize = 5
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Cluster(data, cfg)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFullPipeline_100(b *testing.B)  { benchFullPipeline(b, 100) }
func BenchmarkFullPipeline_500(b *testing.B)  { benchFullPipeline(b, 500) }
func BenchmarkFullPipeline_1000(b *testing.B) { benchFullPipeline(b, 1000) }
