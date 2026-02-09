# hdbscan

Go implementation of [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/) (Hierarchical Density-Based Spatial Clustering of Applications with Noise), ported from the [scikit-learn-contrib/hdbscan](https://github.com/scikit-learn-contrib/hdbscan) Python library.

HDBSCAN finds clusters of varying density without needing to know the number of clusters ahead of time. Points that don't fit any cluster are labeled as noise.

## Install

```bash
go get github.com/TrevorS/hdbscan
```

## Usage

```go
package main

import (
	"fmt"
	"log"

	"github.com/TrevorS/hdbscan"
)

func main() {
	data := [][]float64{
		{1.0, 2.0}, {1.5, 1.8}, {1.2, 2.1},
		{5.0, 8.0}, {5.5, 7.9}, {5.2, 8.3},
		{100.0, 100.0}, // outlier
	}

	cfg := hdbscan.DefaultConfig()
	cfg.MinClusterSize = 3

	result, err := hdbscan.Cluster(data, cfg)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Labels:", result.Labels)           // cluster IDs, -1 = noise
	fmt.Println("Probabilities:", result.Probabilities)  // membership strength [0, 1]
	fmt.Println("Outlier scores:", result.OutlierScores) // GLOSH scores [0, 1]
}
```

### Precomputed distance matrix

If you already have pairwise distances, skip the distance computation step:

```go
// distMatrix is a flat n*n row-major distance matrix
result, err := hdbscan.ClusterPrecomputed(distMatrix, n, cfg)
```

## Configuration

Start with `DefaultConfig()` and override what you need:

| Field | Default | Description |
|-------|---------|-------------|
| `MinClusterSize` | 5 | Smallest group of points considered a cluster |
| `MinSamples` | MinClusterSize | Controls noise sensitivity (higher = more points labeled as noise) |
| `Metric` | Euclidean | Distance function (`EuclideanMetric`, `ManhattanMetric`, `CosineMetric`, `ChebyshevMetric`, `MinkowskiMetric`, or custom via `DistanceFunc`) |
| `ClusterSelectionMethod` | `"eom"` | `"eom"` (Excess of Mass) or `"leaf"` |
| `Alpha` | 1.0 | Distance scaling; values < 1.0 create more aggressive clustering |
| `AllowSingleCluster` | false | Allow all points in one cluster |
| `ClusterSelectionEpsilon` | 0.0 | Minimum distance threshold to prevent over-segmentation |
| `Algorithm` | `"auto"` | MST construction strategy (see [Algorithms](#algorithms) below) |
| `LeafSize` | 40 | Max points per spatial tree leaf node (tree-based algorithms only) |
| `Workers` | `runtime.NumCPU()` | Goroutines for parallel stages in brute-force path (distances, core distances, mutual reachability) |

## Output

`Cluster` returns a `*Result` with:

- **Labels**: cluster ID per point (`-1` = noise)
- **Probabilities**: membership strength in `[0, 1]`
- **OutlierScores**: [GLOSH](https://doi.org/10.1007/978-3-319-49586-6_8) outlier scores in `[0, 1]`
- **Stabilities**: per-cluster stability values
- **CondensedTree**: the condensed cluster hierarchy
- **SingleLinkageTree**: full dendrogram in scipy format

## Algorithms

The `Algorithm` config field selects how the minimum spanning tree is constructed. `"auto"` (default) picks the best strategy based on your metric and data dimensionality:

| Algorithm | Memory | Best for | Notes |
|-----------|--------|----------|-------|
| `"auto"` | varies | General use | Picks `boruvka_kdtree` for standard metrics ≤60 dims, `boruvka_balltree` otherwise, `brute` for custom metrics |
| `"brute"` | O(n²) | Small datasets, custom metrics | Builds full distance matrix; parallelizes distance/core/reachability stages across `Workers` goroutines |
| `"prims_kdtree"` | O(n) | Medium datasets | KD-tree core distances + matrix-free Prim's MST |
| `"prims_balltree"` | O(n) | Medium datasets, higher dims | Ball tree core distances + matrix-free Prim's MST |
| `"boruvka_kdtree"` | O(n) | Large datasets, low dims | Dual-tree Borůvka MST (fastest for ≤60 dims) |
| `"boruvka_balltree"` | O(n) | Large datasets, high dims | Dual-tree Borůvka MST with Ball tree |

Tree-based algorithms require a compatible metric (Euclidean, Manhattan, Chebyshev, or Minkowski). Custom metrics via `DistanceFunc` always use brute force. `ClusterPrecomputed` always uses brute force regardless of this setting.

## Pipeline

```
Input data → Cluster(data, cfg)
  → Algorithm selection           (auto / brute / prims / boruvka)
  → Core distances + MST          (varies by algorithm)
  → Single-linkage dendrogram
  → Condensed tree
  → Stability scores
  → Cluster selection (EOM or leaf)
  → Labels, probabilities, GLOSH outlier scores
```

## Development

```bash
make test    # run tests
make bench   # benchmarks
make lint    # golangci-lint
make race    # tests with race detector
make check   # lint + test + race
```

## References

- Campello, Moulavi, Sander. "Density-Based Clustering Based on Hierarchical Density Estimates" (2013)
- McInnes, Healy, Astels. "hdbscan: Hierarchical density based clustering" (JOSS, 2017)
- Campello et al. "Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection" (2015)

## License

MIT
