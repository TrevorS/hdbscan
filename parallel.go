package hdbscan

import "sync"

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
