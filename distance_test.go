package hdbscan

import (
	"math"
	"testing"
)

const floatTol = 1e-10

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

// --- EuclideanMetric tests ---

func TestEuclideanDistance_IdenticalVectors(t *testing.T) {
	m := EuclideanMetric{}
	a := []float64{1, 2, 3}
	d := m.Distance(a, a)
	if d != 0 {
		t.Errorf("expected 0, got %v", d)
	}
}

func TestEuclideanDistance_ZeroVectors(t *testing.T) {
	m := EuclideanMetric{}
	a := []float64{0, 0, 0}
	b := []float64{0, 0, 0}
	d := m.Distance(a, b)
	if d != 0 {
		t.Errorf("expected 0, got %v", d)
	}
}

func TestEuclideanDistance_UnitVectors(t *testing.T) {
	m := EuclideanMetric{}
	a := []float64{1, 0, 0}
	b := []float64{0, 1, 0}
	// sqrt((1-0)^2 + (0-1)^2 + (0-0)^2) = sqrt(2)
	expected := math.Sqrt(2)
	d := m.Distance(a, b)
	if !almostEqual(d, expected, floatTol) {
		t.Errorf("expected %v, got %v", expected, d)
	}
}

func TestEuclideanDistance_HandComputed(t *testing.T) {
	m := EuclideanMetric{}
	a := []float64{1, 2, 3}
	b := []float64{4, 6, 3}
	// sqrt((4-1)^2 + (6-2)^2 + (3-3)^2) = sqrt(9+16+0) = 5
	d := m.Distance(a, b)
	if !almostEqual(d, 5.0, floatTol) {
		t.Errorf("expected 5.0, got %v", d)
	}
}

func TestEuclideanReducedDistance(t *testing.T) {
	m := EuclideanMetric{}
	a := []float64{1, 2, 3}
	b := []float64{4, 6, 3}
	// squared: 9+16+0 = 25
	rd := m.ReducedDistance(a, b)
	if !almostEqual(rd, 25.0, floatTol) {
		t.Errorf("expected 25.0, got %v", rd)
	}
}

// --- ManhattanMetric tests ---

func TestManhattanDistance_IdenticalVectors(t *testing.T) {
	m := ManhattanMetric{}
	a := []float64{3, 4, 5}
	if d := m.Distance(a, a); d != 0 {
		t.Errorf("expected 0, got %v", d)
	}
}

func TestManhattanDistance_HandComputed(t *testing.T) {
	m := ManhattanMetric{}
	a := []float64{1, 2, 3}
	b := []float64{4, 6, 3}
	// |4-1| + |6-2| + |3-3| = 3+4+0 = 7
	d := m.Distance(a, b)
	if !almostEqual(d, 7.0, floatTol) {
		t.Errorf("expected 7.0, got %v", d)
	}
}

func TestManhattanReducedDistance_EqualsDistance(t *testing.T) {
	m := ManhattanMetric{}
	a := []float64{1, 2, 3}
	b := []float64{4, 6, 3}
	d := m.Distance(a, b)
	rd := m.ReducedDistance(a, b)
	if d != rd {
		t.Errorf("ReducedDistance (%v) != Distance (%v)", rd, d)
	}
}

// --- CosineMetric tests ---

func TestCosineDistance_ParallelVectors(t *testing.T) {
	m := CosineMetric{}
	a := []float64{1, 2, 3}
	b := []float64{2, 4, 6}
	// cosine similarity = 1, distance = 0
	d := m.Distance(a, b)
	if !almostEqual(d, 0.0, floatTol) {
		t.Errorf("expected 0, got %v", d)
	}
}

func TestCosineDistance_OrthogonalVectors(t *testing.T) {
	m := CosineMetric{}
	a := []float64{1, 0}
	b := []float64{0, 1}
	// cosine similarity = 0, distance = 1
	d := m.Distance(a, b)
	if !almostEqual(d, 1.0, floatTol) {
		t.Errorf("expected 1, got %v", d)
	}
}

func TestCosineDistance_IdenticalVectors(t *testing.T) {
	m := CosineMetric{}
	a := []float64{3, 4}
	d := m.Distance(a, a)
	if !almostEqual(d, 0.0, floatTol) {
		t.Errorf("expected 0, got %v", d)
	}
}

func TestCosineDistance_HandComputed(t *testing.T) {
	m := CosineMetric{}
	a := []float64{1, 0, 0}
	b := []float64{1, 1, 0}
	// dot = 1, |a|=1, |b|=sqrt(2)
	// cosine_sim = 1/sqrt(2), distance = 1 - 1/sqrt(2) ~ 0.292893
	expected := 1.0 - 1.0/math.Sqrt(2)
	d := m.Distance(a, b)
	if !almostEqual(d, expected, floatTol) {
		t.Errorf("expected %v, got %v", expected, d)
	}
}

func TestCosineReducedDistance_EqualsDistance(t *testing.T) {
	m := CosineMetric{}
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	d := m.Distance(a, b)
	rd := m.ReducedDistance(a, b)
	if d != rd {
		t.Errorf("ReducedDistance (%v) != Distance (%v)", rd, d)
	}
}

// --- ChebyshevMetric tests ---

func TestChebyshevDistance_IdenticalVectors(t *testing.T) {
	m := ChebyshevMetric{}
	a := []float64{1, 2, 3}
	if d := m.Distance(a, a); d != 0 {
		t.Errorf("expected 0, got %v", d)
	}
}

func TestChebyshevDistance_HandComputed(t *testing.T) {
	m := ChebyshevMetric{}
	a := []float64{1, 2, 3}
	b := []float64{4, 6, 3}
	// max(|4-1|, |6-2|, |3-3|) = max(3, 4, 0) = 4
	d := m.Distance(a, b)
	if !almostEqual(d, 4.0, floatTol) {
		t.Errorf("expected 4.0, got %v", d)
	}
}

func TestChebyshevReducedDistance_EqualsDistance(t *testing.T) {
	m := ChebyshevMetric{}
	a := []float64{1, 2, 3}
	b := []float64{4, 6, 3}
	d := m.Distance(a, b)
	rd := m.ReducedDistance(a, b)
	if d != rd {
		t.Errorf("ReducedDistance (%v) != Distance (%v)", rd, d)
	}
}

// --- MinkowskiMetric tests ---

func TestMinkowskiDistance_IdenticalVectors(t *testing.T) {
	m := MinkowskiMetric{P: 3}
	a := []float64{1, 2, 3}
	if d := m.Distance(a, a); d != 0 {
		t.Errorf("expected 0, got %v", d)
	}
}

func TestMinkowskiDistance_P1_EqualsManhattan(t *testing.T) {
	mink := MinkowskiMetric{P: 1}
	manh := ManhattanMetric{}
	a := []float64{1, 2, 3}
	b := []float64{4, 6, 3}
	dm := mink.Distance(a, b)
	dh := manh.Distance(a, b)
	if !almostEqual(dm, dh, floatTol) {
		t.Errorf("Minkowski P=1 (%v) != Manhattan (%v)", dm, dh)
	}
}

func TestMinkowskiDistance_P2_EqualsEuclidean(t *testing.T) {
	mink := MinkowskiMetric{P: 2}
	eucl := EuclideanMetric{}
	a := []float64{1, 2, 3}
	b := []float64{4, 6, 3}
	dm := mink.Distance(a, b)
	de := eucl.Distance(a, b)
	if !almostEqual(dm, de, floatTol) {
		t.Errorf("Minkowski P=2 (%v) != Euclidean (%v)", dm, de)
	}
}

func TestMinkowskiDistance_P3_HandComputed(t *testing.T) {
	m := MinkowskiMetric{P: 3}
	a := []float64{1, 2, 3}
	b := []float64{4, 6, 3}
	// (|3|^3 + |4|^3 + |0|^3)^(1/3) = (27+64)^(1/3) = 91^(1/3)
	expected := math.Pow(91.0, 1.0/3.0)
	d := m.Distance(a, b)
	if !almostEqual(d, expected, floatTol) {
		t.Errorf("expected %v, got %v", expected, d)
	}
}

func TestMinkowskiDistance_NegativeP_Panics(t *testing.T) {
	m := MinkowskiMetric{P: -1}
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative P, got none")
		}
	}()
	m.Distance(a, b)
}

func TestMinkowskiReducedDistance_P2_IsSquaredEuclidean(t *testing.T) {
	m := MinkowskiMetric{P: 2}
	a := []float64{1, 2, 3}
	b := []float64{4, 6, 3}
	// reduced distance for P=2 is sum(|a[i]-b[i]|^P) = 25
	rd := m.ReducedDistance(a, b)
	if !almostEqual(rd, 25.0, floatTol) {
		t.Errorf("expected 25.0, got %v", rd)
	}
}

// --- DistanceFunc adapter tests ---

func TestDistanceFunc_Adapter(t *testing.T) {
	fn := DistanceFunc(func(a, b []float64) float64 {
		sum := 0.0
		for i := range a {
			sum += math.Abs(a[i] - b[i])
		}
		return sum
	})
	a := []float64{1, 2, 3}
	b := []float64{4, 6, 3}

	d := fn.Distance(a, b)
	if !almostEqual(d, 7.0, floatTol) {
		t.Errorf("expected 7.0, got %v", d)
	}

	rd := fn.ReducedDistance(a, b)
	if d != rd {
		t.Errorf("ReducedDistance (%v) != Distance (%v) for DistanceFunc adapter", rd, d)
	}
}

func TestDistanceFunc_SatisfiesInterface(t *testing.T) {
	fn := DistanceFunc(func(a, b []float64) float64 { return 0 })
	var _ DistanceMetric = fn // compile-time check
}

// --- Zero vector tests for all metrics ---

func TestAllMetrics_ZeroVectors(t *testing.T) {
	metrics := map[string]DistanceMetric{
		"euclidean":  EuclideanMetric{},
		"manhattan":  ManhattanMetric{},
		"cosine":     CosineMetric{},
		"chebyshev":  ChebyshevMetric{},
		"minkowski3": MinkowskiMetric{P: 3},
	}
	zero := []float64{0, 0, 0}

	for name, m := range metrics {
		d := m.Distance(zero, zero)
		// Cosine of two zero vectors is NaN (0/0), which is a special case.
		// All others should be 0.
		if name == "cosine" {
			if !math.IsNaN(d) {
				t.Errorf("%s: expected NaN for zero vectors, got %v", name, d)
			}
		} else {
			if d != 0 {
				t.Errorf("%s: expected 0 for zero vectors, got %v", name, d)
			}
		}
	}
}

// --- ComputePairwiseDistances tests ---

func TestComputePairwiseDistances_3Points(t *testing.T) {
	// Points: (0,0), (3,0), (0,4)
	data := []float64{
		0, 0,
		3, 0,
		0, 4,
	}
	n, dims := 3, 2

	dist := ComputePairwiseDistances(data, n, dims, EuclideanMetric{})

	if len(dist) != 9 {
		t.Fatalf("expected length 9, got %d", len(dist))
	}

	// Expected: 3-4-5 triangle
	expected := []float64{
		0, 3, 4,
		3, 0, 5,
		4, 5, 0,
	}

	for i := 0; i < 9; i++ {
		if !almostEqual(dist[i], expected[i], floatTol) {
			row, col := i/n, i%n
			t.Errorf("dist[%d,%d] = %v, expected %v", row, col, dist[i], expected[i])
		}
	}
}

func TestComputePairwiseDistances_Symmetry(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	n, dims := 4, 2

	dist := ComputePairwiseDistances(data, n, dims, EuclideanMetric{})

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !almostEqual(dist[i*n+j], dist[j*n+i], floatTol) {
				t.Errorf("dist[%d,%d]=%v != dist[%d,%d]=%v", i, j, dist[i*n+j], j, i, dist[j*n+i])
			}
		}
	}
}

func TestComputePairwiseDistances_DiagonalZero(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	n, dims := 3, 2

	dist := ComputePairwiseDistances(data, n, dims, EuclideanMetric{})

	for i := 0; i < n; i++ {
		if dist[i*n+i] != 0 {
			t.Errorf("diagonal dist[%d,%d] = %v, expected 0", i, i, dist[i*n+i])
		}
	}
}

func TestComputePairwiseDistances_ManhattanMetric(t *testing.T) {
	data := []float64{0, 0, 3, 4}
	n, dims := 2, 2

	dist := ComputePairwiseDistances(data, n, dims, ManhattanMetric{})

	// d(0,1) = |3|+|4| = 7
	if !almostEqual(dist[0*n+1], 7.0, floatTol) {
		t.Errorf("expected 7.0, got %v", dist[0*n+1])
	}
	if !almostEqual(dist[1*n+0], 7.0, floatTol) {
		t.Errorf("expected 7.0, got %v", dist[1*n+0])
	}
}
