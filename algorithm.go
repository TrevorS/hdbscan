package hdbscan

import "fmt"

// KDTreeValidMetric reports whether the metric supports KD-tree acceleration.
// KD-trees require metrics that decompose along coordinate axes:
// Euclidean, Manhattan, Chebyshev, Minkowski.
func KDTreeValidMetric(m DistanceMetric) bool {
	switch m.(type) {
	case EuclideanMetric, ManhattanMetric, ChebyshevMetric, MinkowskiMetric:
		return true
	default:
		return false
	}
}

// BallTreeValidMetric reports whether the metric supports Ball tree acceleration.
// Ball trees work with any metric that satisfies the triangle inequality.
// Currently accepts the same set as KD-tree; future metrics (e.g. Haversine)
// can be added here without also adding them to KDTreeValidMetric.
func BallTreeValidMetric(m DistanceMetric) bool {
	switch m.(type) {
	case EuclideanMetric, ManhattanMetric, ChebyshevMetric, MinkowskiMetric:
		return true
	default:
		return false
	}
}

// selectAlgorithm resolves AlgorithmAuto into a concrete algorithm choice
// based on the metric and data dimensionality, and validates that user-forced
// algorithm choices are compatible with the metric.
func selectAlgorithm(cfg Config, n, dims int) (Algorithm, error) {
	algo := cfg.Algorithm

	if algo == AlgorithmAuto {
		if !BallTreeValidMetric(cfg.Metric) {
			return AlgorithmBrute, nil
		}
		if KDTreeValidMetric(cfg.Metric) && dims <= 60 {
			return AlgorithmBoruvkaKDTree, nil
		}
		return AlgorithmBoruvkaBalltree, nil
	}

	// Validate user-forced choices.
	switch algo {
	case AlgorithmPrimsKDTree, AlgorithmBoruvkaKDTree:
		if !KDTreeValidMetric(cfg.Metric) {
			return "", fmt.Errorf("hdbscan: metric %T is not supported by KD-tree algorithms", cfg.Metric)
		}
	case AlgorithmPrimsBalltree, AlgorithmBoruvkaBalltree:
		if !BallTreeValidMetric(cfg.Metric) {
			return "", fmt.Errorf("hdbscan: metric %T is not supported by Ball tree algorithms", cfg.Metric)
		}
	}

	return algo, nil
}
