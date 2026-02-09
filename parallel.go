package hdbscan

import (
	"sort"
	"sync"
)

// ComputePairwiseDistancesParallel computes the full n×n distance matrix using
// multiple goroutines. data is flat row-major with n rows and dims columns.
// numWorkers controls the degree of parallelism; if <= 1, it falls back to
// single-threaded ComputePairwiseDistances.
//
// The result is bitwise identical to ComputePairwiseDistances: a flat []float64
// of length n×n in row-major order.
func ComputePairwiseDistancesParallel(data []float64, n, dims int, metric DistanceMetric, numWorkers int) []float64 {
	if numWorkers <= 1 || n <= 1 {
		return ComputePairwiseDistances(data, n, dims, metric)
	}

	result := make([]float64, n*n)

	// Split rows across workers. Each worker handles a contiguous range of
	// "source" rows and computes dist(i,j) for all j > i in that range.
	// Since row ranges don't overlap, no synchronization is needed for writes.
	var wg sync.WaitGroup

	rowsPerWorker := (n + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > n {
			endRow = n
		}
		if startRow >= n {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := i + 1; j < n; j++ {
					d := metric.Distance(data[i*dims:(i+1)*dims], data[j*dims:(j+1)*dims])
					result[i*n+j] = d
					result[j*n+i] = d
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()
	return result
}

// ComputeCoreDistancesParallel computes core distances using multiple goroutines.
// Each worker handles a contiguous range of points independently.
// Falls back to sequential ComputeCoreDistances if numWorkers <= 1.
func ComputeCoreDistancesParallel(distMatrix []float64, n, minSamples, numWorkers int) []float64 {
	if numWorkers <= 1 || n <= 1 {
		return ComputeCoreDistances(distMatrix, n, minSamples)
	}

	minSamples = min(minSamples, n-1)
	minSamples = max(minSamples, 0)

	core := make([]float64, n)
	if minSamples == 0 {
		return core
	}

	var wg sync.WaitGroup
	rowsPerWorker := (n + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > n {
			endRow = n
		}
		if startRow >= n {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			neighbors := make([]float64, n-1)
			for i := start; i < end; i++ {
				k := 0
				for j := 0; j < n; j++ {
					if j != i {
						neighbors[k] = distMatrix[i*n+j]
						k++
					}
				}
				sort.Float64s(neighbors)
				core[i] = neighbors[minSamples-1]
			}
		}(startRow, endRow)
	}

	wg.Wait()
	return core
}

// MutualReachabilityParallel computes the mutual reachability distance matrix
// using multiple goroutines. Each worker handles a contiguous range of rows.
// Falls back to sequential MutualReachability if numWorkers <= 1.
func MutualReachabilityParallel(distMatrix, coreDistances []float64, n int, alpha float64, numWorkers int) []float64 {
	if numWorkers <= 1 || n <= 1 {
		return MutualReachability(distMatrix, coreDistances, n, alpha)
	}

	result := make([]float64, n*n)

	var wg sync.WaitGroup
	rowsPerWorker := (n + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > n {
			endRow = n
		}
		if startRow >= n {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				ci := coreDistances[i]
				for j := 0; j < n; j++ {
					d := distMatrix[i*n+j]
					if alpha != 1.0 {
						d /= alpha
					}
					result[i*n+j] = max(d, ci, coreDistances[j])
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()
	return result
}
