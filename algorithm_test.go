package hdbscan

import (
	"testing"
)

func TestSelectAlgorithmAuto(t *testing.T) {
	tests := []struct {
		name     string
		metric   DistanceMetric
		dims     int
		expected Algorithm
	}{
		{
			name:     "euclidean low dim → boruvka_kdtree",
			metric:   EuclideanMetric{},
			dims:     3,
			expected: AlgorithmBoruvkaKDTree,
		},
		{
			name:     "euclidean dim=60 → boruvka_kdtree",
			metric:   EuclideanMetric{},
			dims:     60,
			expected: AlgorithmBoruvkaKDTree,
		},
		{
			name:     "euclidean dim=61 → boruvka_balltree",
			metric:   EuclideanMetric{},
			dims:     61,
			expected: AlgorithmBoruvkaBalltree,
		},
		{
			name:     "manhattan low dim → boruvka_kdtree",
			metric:   ManhattanMetric{},
			dims:     10,
			expected: AlgorithmBoruvkaKDTree,
		},
		{
			name:     "cosine → brute (not ball-valid)",
			metric:   CosineMetric{},
			dims:     5,
			expected: AlgorithmBrute,
		},
		{
			name:     "custom DistanceFunc → brute",
			metric:   DistanceFunc(func(a, b []float64) float64 { return 0 }),
			dims:     2,
			expected: AlgorithmBrute,
		},
		{
			name:     "minkowski low dim → boruvka_kdtree",
			metric:   MinkowskiMetric{P: 3},
			dims:     5,
			expected: AlgorithmBoruvkaKDTree,
		},
		{
			name:     "chebyshev high dim → boruvka_balltree",
			metric:   ChebyshevMetric{},
			dims:     100,
			expected: AlgorithmBoruvkaBalltree,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{
				Algorithm: AlgorithmAuto,
				Metric:    tc.metric,
			}
			got, err := selectAlgorithm(cfg, 100, tc.dims)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.expected {
				t.Errorf("expected %q, got %q", tc.expected, got)
			}
		})
	}
}

func TestSelectAlgorithmExplicit(t *testing.T) {
	tests := []struct {
		name    string
		algo    Algorithm
		metric  DistanceMetric
		wantErr bool
	}{
		{
			name:    "brute with any metric",
			algo:    AlgorithmBrute,
			metric:  CosineMetric{},
			wantErr: false,
		},
		{
			name:    "boruvka_kdtree with euclidean",
			algo:    AlgorithmBoruvkaKDTree,
			metric:  EuclideanMetric{},
			wantErr: false,
		},
		{
			name:    "boruvka_kdtree with cosine → error",
			algo:    AlgorithmBoruvkaKDTree,
			metric:  CosineMetric{},
			wantErr: true,
		},
		{
			name:    "prims_kdtree with cosine → error",
			algo:    AlgorithmPrimsKDTree,
			metric:  CosineMetric{},
			wantErr: true,
		},
		{
			name:    "boruvka_balltree with euclidean",
			algo:    AlgorithmBoruvkaBalltree,
			metric:  EuclideanMetric{},
			wantErr: false,
		},
		{
			name:    "boruvka_balltree with cosine → error",
			algo:    AlgorithmBoruvkaBalltree,
			metric:  CosineMetric{},
			wantErr: true,
		},
		{
			name:    "prims_balltree with custom func → error",
			algo:    AlgorithmPrimsBalltree,
			metric:  DistanceFunc(func(a, b []float64) float64 { return 0 }),
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{
				Algorithm: tc.algo,
				Metric:    tc.metric,
			}
			_, err := selectAlgorithm(cfg, 100, 10)
			if tc.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestKDTreeValidMetric(t *testing.T) {
	valid := []DistanceMetric{
		EuclideanMetric{},
		ManhattanMetric{},
		ChebyshevMetric{},
		MinkowskiMetric{P: 2},
	}
	for _, m := range valid {
		if !KDTreeValidMetric(m) {
			t.Errorf("expected %T to be valid for KD-tree", m)
		}
	}

	invalid := []DistanceMetric{
		CosineMetric{},
		DistanceFunc(func(a, b []float64) float64 { return 0 }),
	}
	for _, m := range invalid {
		if KDTreeValidMetric(m) {
			t.Errorf("expected %T to be invalid for KD-tree", m)
		}
	}
}

func TestBallTreeValidMetric(t *testing.T) {
	valid := []DistanceMetric{
		EuclideanMetric{},
		ManhattanMetric{},
		ChebyshevMetric{},
		MinkowskiMetric{P: 2},
	}
	for _, m := range valid {
		if !BallTreeValidMetric(m) {
			t.Errorf("expected %T to be valid for Ball tree", m)
		}
	}

	invalid := []DistanceMetric{
		CosineMetric{},
		DistanceFunc(func(a, b []float64) float64 { return 0 }),
	}
	for _, m := range invalid {
		if BallTreeValidMetric(m) {
			t.Errorf("expected %T to be invalid for Ball tree", m)
		}
	}
}
