# Parameter Selection Guide

This guide provides detailed recommendations for tuning MuH-MDS parameters to optimize embedding quality and computational efficiency. For basic usage examples, see [Example Usage Guide](2-Example%20Usage.md).

## Quick Reference Tables

### Core Parameters by Dataset Size

| Dataset Size | Number of Clusters | Dimensions | Max Cluster Size | Min Cluster Size | 
|--------------|-------------------|------------|------------------|------------------|
| < 1,000 | 10-100 | 2-3 | 200-400 | 1 |
| 1,000-10,000 | 100-200 | 3-5 | 200-500 | 3 |
| 10,000-50,000 | 200-300 | 3-10 | 500-1000 | 3 |
| > 50,000 | 300-800 | 5-10 | 400-500 | 3-5 |

**Note**: The optimal number of clusters for computational efficiency is approximately $n^{2/3}$, where $n$ is the dataset size.

### Number of Neighbors by Dimension

| Embedding Dimension | Recommended Neighbors |
|---------------------|----------------------|
| 2-3 | 30 |
| 4-10 | 50 |
| 10-20 | 75 |

**Rule of thumb**: Use at least `dimension × 3` neighbors for high-dimensional embeddings.

---

## Boolean Flags

### `--save-matrix`
**Purpose**: Save embedding coordinate matrices to disk.

**Recommendation**: Always include this flag (default: False).

**Usage**: 
```bash
python Multiscale_hmds.py ... --save-matrix
```

### `--compute-metrics`
**Purpose**: Calculate embedding quality metrics (Qlocal, Qglobal, correlation).

**Recommendation**: Include for quality assessment, omit for very large datasets.
- Computational complexity: $O(N^2)$, can be time-consuming when $N > 40,000$
- Skip for initial exploratory runs on large datasets

**Usage**:
```bash
# Include metrics (default behavior)
python Multiscale_hmds.py ... --compute-metrics

# Skip metrics for speed
python Multiscale_hmds.py ...  # omit flag
```

### `--map-outlier`
**Purpose**: Remap excluded outliers back to embedding space after global-local embedding.

**Recommendation**: Include when `--min-cluster-size > 1`.
- Outliers (clusters smaller than min-cluster-size) are initially excluded
- This flag remaps them based on $k$-nearest neighbors in the embedded space
- The $k$ value is set by `--outlier-neighbors`

**Usage**:
```bash
python Multiscale_hmds.py ... --min-cluster-size 3 --map-outlier --outlier-neighbors 10
```

---

## Detailed Parameter Guidance

### Number of Clusters (`--n-clusters` / `--thresholds`)

**What it controls**: The number of clusters that the data is divided into. 

**Impact on quality**:
- Embedding quality metrics are sensitive to cluster count only when very small (< 10 clusters)

**Impact on runtime**:
- Computational time varies significantly with cluster count
- **Optimal value**: Approximately $n^{2/3}$ for best global-local embedding time trade-off

**Feature matrix workflow** (K-means):
```bash
python Generate_cls.py --featmat MyData.txt --n-clusters 150
```
- Specify exact number of clusters
- For dataset with 10,000 samples: $10000^{2/3} \approx 215$ clusters

**Distance matrix workflow** (Agglomerative clustering):
```bash
python Generate_cls.py --distmat MyData.txt --thresholds 0.4
```
- Specify distance threshold instead of cluster count
- Start with `threshold = 0.4` and examine resulting cluster count
- Iterate to achieve approximately $n^{2/3}$ clusters

**Recommendations**:
- **< 1,000 samples**: 10-100 clusters
- **1,000-10,000 samples**: 100-200 clusters (aim for $n^{2/3}$)
- **10,000-50,000 samples**: 200-300 clusters
- **> 50,000 samples**: 300-800 clusters

<p align="center">
  <img src="images/k_quality.png" width="400">
</p>
<p align="center">
  <em>Figure 1a: Embedding quality (Qlocal and Qglobal) vs. number of clusters. Quality is insensitive to cluster count when sufficient clusters are used.</em>
</p>

<p align="center">
  <img src="images/k_time.png" width="400">
</p>
<p align="center">
  <em>Figure 1b: Computational time vs. number of clusters. Optimal cluster count (n<sup>2/3</sup>) minimizes runtime while maintaining quality.</em>
</p>

---

### Number of Neighbors (`--neighbors`)

**What it controls**: The number of nearest neighbors used to constrain local geometry during embedding.

**Impact on quality**:
- Sensitivity depends on embedding dimension:
  - **Low dimensions (2-3D)**: Quality relatively insensitive to neighbor count
  - **High dimensions (>5D)**: More neighbors required as constraints
- More neighbors are needed in higher dimensions to uniquely determine point positions

**Impact on local vs. global structure**:
- **Fewer neighbors (5-20)**: Emphasizes fine-grained local structure
  - Best for trajectory data, manifolds, gradual transitions
- **More neighbors (50-100)**: Emphasizes broad global structure
  - Best for hierarchical data, discrete clusters, tree structures

**Recommendations**:
- **2D-3D embeddings**: 20-30 neighbors (sufficient for visualization)
- **4D-10D embeddings**: 50 neighbors
- **10D-20D embeddings**: 75 neighbors
- **General rule**: Use at least `dimension × 3` neighbors
- **Alternative rule**: Use 5-10% of dataset size

**Example**:
```bash
# Low-dimensional visualization (emphasize local structure)
python Multiscale_hmds.py ... --neighbors 30 --dimension 3

# High-dimensional embedding (emphasize global structure)
python Multiscale_hmds.py ... --neighbors 75 --dimension 15
```

<p align="center">
  <img src="images/alpha_quality.png" width="600">
</p>
<p align="center">
  <em>Figure 2: Embedding quality vs. number of neighbors for different dimensions. Higher dimensions require more neighbors, while 2D-3D embeddings are relatively insensitive.</em>
</p>

---

### Embedding Dimension (`--dimension`)

**What it controls**: The dimensionality of the hyperbolic embedding space.

**Impact on quality**:
- **Qlocal** (local structure): Improves monotonically with dimension
  - More dimensions = more capacity to capture fine details
- **Qglobal** (global structure): Saturates around D = 4
  - Hyperbolic geometry captures most global information in low dimensions
- **Correlation**: Saturates around D = 4
  - Distance preservation plateaus quickly

**Key insight**: Hyperbolic space is remarkably efficient at representing hierarchical and global structure in just 2-3 dimensions, unlike Euclidean methods which often require many more dimensions.

**Trade-offs**:
- **2D**: Best for visualization, easy interpretation, fast computation
- **3D**: Optimal balance between visualization and detail preservation
- **5D-10D**: Capture complex structures, require dimensionality reduction for visualization
- **>10D**: Maximum quality, but visualization requires projection

**Recommendations**:
- **For visualization**: Use 2D or 3D (preferred: 3D)
- **For downstream analysis**: Use 3D-5D for best quality
- **For maximum quality**: Use 5D-10D, accept visualization complexity

**Example**:
```bash
# Visualization-focused
python Multiscale_hmds.py ... --dimension 2 --neighbors 30

# Quality-focused
python Multiscale_hmds.py ... --dimension 5 --neighbors 50
```

<p align="center">
  <img src="images/dimension.png" width="400">
</p>
<p align="center">
  <em>Figure 3: Embedding quality vs. dimension. Qlocal continues improving, while Qglobal saturates at D=4, demonstrating hyperbolic space's efficiency at capturing global structure.</em>
</p>

---

### Maximum Cluster Size (`--max-cluster-size`)

**What it controls**: The maximum allowed size for clusters before subdivision during refinement.

**Purpose**: Prevent extremely large clusters that would slow down local embedding steps.

**How it works**:
- During hierarchical refinement, clusters exceeding this size are subdivided
- Creates a more balanced cluster size distribution
- Significantly reduces embedding time for large clusters

**Impact on runtime**:
- **Too small (~100)**: Over-subdivision increases global embedding time
- **Too large (~1000)**: Large clusters slow down local embedding steps
- **Optimal (~400)**: Balances local and global computational costs

**Impact on quality**:
- Minimal quality impact when set appropriately

**Recommendations**:
- **Default**: 400 (works well for most datasets)
- **Always use**: `--max-cluster-size ≤ 400` to avoid clusters > 500
- **Large datasets (>50,000)**: Can use 400-500
- **Small datasets (<3,000)**: Can use 200-300

**Example**:
```bash
python Multiscale_hmds.py ... --max-cluster-size 400
```

<p align="center">
  <img src="images/max-min.png" width="700">
</p>
<p align="center">
  <em>Figure 4: Impact of cluster size control on embedding quality and runtime. Appropriate bounds on cluster sizes optimize both metrics.</em>
</p>

---


### Min Cluster Size (`--min-cluster-size`)

For very small clusters, for example, those with only 1-2 samples in a cluster, we apply an “exclude and remap” procedure to reduce the total number of clusters in the global embedding step. Specifically, small clusters are excluded from the global-local embedding process and are mapped to the embedding space only after all other points have been embedded. 

The motivation for the “exclude and remap” strategy is that very small clusters provide poor constraints and can degrade centroid-based global structure. By excluding clusters that are too small, the algorithm focuses on capturing the geometry of the majority of points during the embedding step, leading to a more precise representation of the global geometry.

- **No filtering (min-cluster-size = 1)**: None of the clusters are excluded, may increase global embedding time significantly and reduce embedding quality. 
- **Filter very small clusters (min-cluster-size = 3)**: Fair control of filtering, benefits both quality and embedding time
- **Filter medium size clusters (min-cluster-size = 10)**: May increase time in remapping and lose information in those clusters filtered
- For small datasets (size < 3,000) it is acceptable to use `min-cluster-size = 1`, for large datasets we suggest using `min-cluster-size = 3`.


## Troubleshooting Common Issues

### 1. Low Quality Metrics

**Problem**: Qlocal < 0.5 or Qglobal < 0.4

**Diagnosis**:
- **Low Qlocal**: Local neighborhood structure is not well-preserved
- **Low Qglobal**: Global distance relationships are distorted

**Solutions**:
- **For low Qlocal**: 
  - Increase `--neighbors` to capture more local context
  - Decrease `--min-cluster-size` to avoid filtering too many outliers
  - Try higher dimensions (3D or 5D instead of 2D)
- **For low Qglobal**: 
  - Increase `--neighbors` to emphasize global structure
  - Adjust cluster count closer to $n^{2/3}$
  - Check if your data has inherent hierarchical structure
- **Data quality checks**:
  - Remove extreme outliers
  - Normalize features (zero mean, unit variance)
  - Verify distance matrix is valid (symmetric, non-negative)

### 2. Slow Runtime

**Problem**: Embedding takes > 30 minutes

**Diagnosis**: Check which step is slow (clustering, global embedding, or local embedding)

**Solutions**:
- **Optimize cluster count**: Aim for $n^{2/3}$ clusters
  - Too few clusters: Slow local embedding
  - Too many clusters: Slow global embedding
- **Control cluster sizes**:
  - Set `--max-cluster-size 400` to subdivide large clusters
  - Set `--min-cluster-size 3` to exclude small clusters
- **Reduce neighbors**: Use 20-30 for 2D/3D instead of 50+
- **Skip metrics**: Omit `--compute-metrics` flag (saves $O(N^2)$ time)
- **For very large datasets**: Use the large-scale variant in `MuH-MDS/large_scale_hmds.py`

### 3. Poor Visualization

**Problem**: All points clustered in center or at boundary

**Diagnosis**:
- Points at center: Insufficient hierarchy or flat structure
- Points at boundary: Numerical issues or extreme distances

**Solutions**:
- **Convert coordinates**: Use `to_native()` function from the example notebook
  ```python
  native_coords = to_native(poincare_coords)
  ```
- **Check embedding quality**: Inspect metrics file - low metrics indicate failed embedding
- **Try higher dimensions**: Use 3D instead of 2D for complex structures
- **Adjust parameters**:
  - Increase `--neighbors` for more constraints
  - Adjust cluster count
  - Try different `--max-cluster-size`
- **Data preprocessing**:
  - Apply PCA if features > 100 (keep 50-100 components)
  - Standardize features
  - Remove duplicate samples

### 4. Numerical Errors

**Problem**: Stan model fails or returns NaN values

**Solutions**:
- **For distance matrices**:
  - Verify all values are non-negative
  - Check for infinite values
  - Ensure matrix is symmetric
  - Verify triangle inequality holds (approximately)
- **For feature matrices**:
  - Remove NaN or infinite values
  - Standardize features (critical for numerical stability)
  - Check for constant features (zero variance)
- **Reduce neighbors**: Try smaller `--neighbors` value
- **Check Stan installation**: Ensure CmdStanPy 1.1.0 and models are compiled

### 5. Memory Errors

**Problem**: Out of memory during embedding

**Solutions**:
- Reduce `--neighbors` (major memory reduction)
- Increase `--min-cluster-size` to exclude more small clusters
- Increase `--max-cluster-size` to reduce hierarchy depth
- Remove `--save-matrix` flag if intermediate matrices not needed
- For distance matrices: Use feature matrix workflow instead (more memory-efficient)
- Consider using a machine with more RAM or the large-scale variant

---

## Summary

### Quick Recommendations by Use Case

| Use Case | Dimension | Neighbors | Clusters | Max Size | Min Size |
|----------|-----------|-----------|----------|----------|----------|
| **Quick visualization** | 2 | 30 | $n^{2/3}$ | 400 | 1 |
| **Quality embedding** | 3-5 | 50 | $n^{2/3}$ | 400 | 3 |
| **Large dataset (>50k)** | 3 | 50-100 | $n^{2/3}$ | 500 | 3-5 |
| **Small dataset (<1k)** | 2-3 | 10-20 | 20-50 | 300 | 1 |
| **Hierarchical data** | 2-3 | 50-100 | $n^{2/3}$ | 400 | 3 |
| **Trajectory data** | 2-3 | 10-30 | $n^{2/3}$ | 400 | 1 |

### Key Takeaways

1. **Always aim for** $n^{2/3}$ **clusters** for optimal runtime
2. **Use more neighbors** (50-100) for hierarchical data, fewer (10-30) for trajectories
3. **3D embeddings** provide the best balance between quality and visualization
4. **Set max-cluster-size ≤ 400** to avoid slow local embedding
5. **Set min-cluster-size = 3** for large datasets (>3,000 samples)
6. **Always use `--save-matrix`** to enable visualization and analysis
7. **Hyperbolic space is efficient**: 2-3D captures most hierarchical structure

### Further Reading

- **Example usage**: See [2-Example Usage.md](2-Example%20Usage.md) for hands-on tutorials
- **Algorithm intuition**: See [1-Overview.md](1-Overview.md) for conceptual background
- **Interactive examples**: Run [example-usage.ipynb](example-usage.ipynb) for working code

