package hdbscan

import "math"

// DistanceMetric provides distance computation with optional reduced distance
// for tree-pruning optimizations (e.g., squared Euclidean skips sqrt).
type DistanceMetric interface {
	Distance(a, b []float64) float64
	ReducedDistance(a, b []float64) float64
}

// DistanceFunc adapts a plain function into a DistanceMetric.
// ReducedDistance delegates to the same function.
type DistanceFunc func(a, b []float64) float64

func (f DistanceFunc) Distance(a, b []float64) float64        { return f(a, b) }
func (f DistanceFunc) ReducedDistance(a, b []float64) float64 { return f(a, b) }

// EuclideanMetric computes the Euclidean (L2) distance.
// ReducedDistance returns squared Euclidean distance (skips sqrt).
type EuclideanMetric struct{}

func (EuclideanMetric) Distance(a, b []float64) float64 {
	return math.Sqrt(euclideanSumOfSquares(a, b))
}

func (EuclideanMetric) ReducedDistance(a, b []float64) float64 {
	return euclideanSumOfSquares(a, b)
}

func euclideanSumOfSquares(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// ManhattanMetric computes the Manhattan (L1 / city-block) distance.
type ManhattanMetric struct{}

func (ManhattanMetric) Distance(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += math.Abs(a[i] - b[i])
	}
	return sum
}

func (m ManhattanMetric) ReducedDistance(a, b []float64) float64 { return m.Distance(a, b) }

// CosineMetric computes the cosine distance: 1 - cosine_similarity.
// For two zero vectors, the result is NaN (0/0).
type CosineMetric struct{}

func (CosineMetric) Distance(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	return 1.0 - dot/math.Sqrt(normA*normB)
}

func (m CosineMetric) ReducedDistance(a, b []float64) float64 { return m.Distance(a, b) }

// ChebyshevMetric computes the Chebyshev (L-infinity) distance.
type ChebyshevMetric struct{}

func (ChebyshevMetric) Distance(a, b []float64) float64 {
	var maxVal float64
	for i := range a {
		if v := math.Abs(a[i] - b[i]); v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

func (m ChebyshevMetric) ReducedDistance(a, b []float64) float64 { return m.Distance(a, b) }

// MinkowskiMetric computes the Minkowski distance parameterized by P.
// P must be >= 1. Panics if P < 1.
// ReducedDistance returns sum(|a[i]-b[i]|^P) without the final root.
type MinkowskiMetric struct {
	P float64
}

func (m MinkowskiMetric) Distance(a, b []float64) float64 {
	return math.Pow(m.rawSum(a, b), 1.0/m.P)
}

func (m MinkowskiMetric) ReducedDistance(a, b []float64) float64 {
	return m.rawSum(a, b)
}

func (m MinkowskiMetric) rawSum(a, b []float64) float64 {
	if m.P < 1 {
		panic("MinkowskiMetric: P must be >= 1")
	}
	var sum float64
	for i := range a {
		sum += math.Pow(math.Abs(a[i]-b[i]), m.P)
	}
	return sum
}

// ComputePairwiseDistances computes the full n*n distance matrix.
// data is flat row-major with n rows and dims columns.
// Returns flat []float64 of length n*n.
func ComputePairwiseDistances(data []float64, n, dims int, metric DistanceMetric) []float64 {
	result := make([]float64, n*n)

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			d := metric.Distance(data[i*dims:(i+1)*dims], data[j*dims:(j+1)*dims])
			result[i*n+j] = d
			result[j*n+i] = d
		}
	}

	return result
}
