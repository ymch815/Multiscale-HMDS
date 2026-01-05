# Execution Plan: Example Usage Tutorial

## Overview
Create comprehensive tutorial materials demonstrating MHMDS usage through two complete examples:
1. **Toggle Switch** (feature matrix → 2D hyperbolic)
2. **WordNet Mammals** (distance matrix → 2D hyperbolic)

## Target Audience
Researchers with basic Python knowledge who want to:
- Run MHMDS on their own data
- Understand the complete workflow from data to visualization
- Learn parameter selection through concrete examples

---

## File 1: `2-Example Usage.md`

### Proposed Structure (Revised)

#### 1. Introduction
- Brief overview of the two examples
- What readers will learn
- Prerequisites (Python environment, dependencies)
- Link to installation instructions in README

#### 2. Example 1: Toggle Switch (Feature Matrix)

**2.1 Understanding the Dataset**
- Brief description: synthetic gene regulatory network data
- Data dimensions and characteristics
- Why this is a good feature matrix example
- Hierarchical structure in the data

**2.2 Data Preparation**
- Location: `data/FeatMat_ToggleSwitch/FeatMat_ToggleSwitch.txt`
- Format explanation: rows = samples, columns = features
- Quick data inspection (shape, preview)

**2.3 Step 1: Clustering**
```bash
cd MultiscalehMDS_feature
python Generate_cls.py --featmat FeatMat_ToggleSwitch.txt --n-clusters 20
```
- What this command does
- Parameter explanation: `--n-clusters 20` (single-layer hierarchy)
- Output: `cls_20` file
- How to inspect cluster assignments

**2.4 Step 2: Embedding**
```bash
python Multiscale_hmds.py --featmat FeatMat_ToggleSwitch.txt --clusters cls_20 \
       --neighbors 10 --dimension 2 --min-cluster-size 1 --max-cluster-size 400 \
       --save-matrix --compute-metrics --outlier-neighbors 10
```
- Parameter breakdown with rationale
- Expected runtime
- Output files location: `./test/FeatMat_ToggleSwitch/`
- Output file naming convention

**2.5 Understanding the Output**
- Coordinate files: `*_coords_20.txt` (centroids), `*_coords_XXX.txt` (individuals)
- Index file: `*_inds_after_prep.txt` (ordering information)
- Metrics file: `*_metrics.csv` (quality measures)

**2.6 Visualizing Results**
- Brief mention: see companion Jupyter notebook
- Key visualization: scatter plot in Poincaré disk
- Interpreting the hyperbolic embedding

#### 3. Example 2: WordNet Mammals (Distance Matrix)

**3.1 Understanding the Dataset**
- Brief description: hierarchical taxonomy from WordNet
- Pre-computed distance matrix (graph distances)
- Why this demonstrates hierarchical structure well
- Number of mammal species in dataset

**3.2 Data Preparation**
- Location: `data/DistMat_mammalwords/DistMat_mammalwords.txt`
- Format explanation: symmetric distance matrix
- Relationship to graph structure

**3.3 Step 1: Clustering**
```bash
cd MultiscalehMDS_distance
python Generate_cls.py --distmat DistMat_mammalwords.txt --thresholds 0.4
```
- Difference from feature matrix clustering (agglomerative vs K-means)
- Parameter explanation: `--thresholds 0.4`
- Output: `cls_0.4` file

**3.4 Step 2: Embedding**
```bash
python Multiscale_hmds.py --distmat DistMat_mammalwords.txt --clusters cls_0.4 \
       --neighbors 60 --dimension 2 --min-cluster-size 1 --max-cluster-size 300 \
       --save-matrix --compute-metrics --outlier-neighbors 10
```
- Parameter differences from Toggle Switch
- Why more neighbors (60 vs 10)?
- Output files location: `./test/DistMat_mammalwords/`

**3.5 Understanding the Output**
- Same file structure as Example 1
- How to interpret hierarchical embedding
- Quality metrics interpretation

**3.6 Visualizing Results**
- Network visualization in hyperbolic space
- Hierarchical levels shown by distance from origin
- Node labels and tree structure
- Interpreting radial distance as hierarchy depth

#### 4. Working with Your Own Data

**4.1 Feature Matrix Requirements**
- Format specifications
- Recommended preprocessing
- When to use feature matrices

**4.2 Distance Matrix Requirements**
- Format specifications
- Valid distance metrics
- When to use distance matrices

**4.3 Quick Checklist**
- Data format validation
- Directory structure
- Parameter selection guidelines

#### 5. Next Steps
- Link to Parameter Guide for detailed tuning
- Link to Overview for conceptual understanding
- Tips for exploring results

---

## File 2: `example-usage.ipynb`

### Proposed Structure

#### Cell 1: Introduction (Markdown)
- Notebook purpose
- Two examples overview
- Prerequisites check

#### Cell 2: Setup and Imports (Python)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../')
import MHMDS.embed_funs as emb
```

#### Cell 3: Helper Functions (Python)
```python
def to_native(coords):
    """Convert Poincaré coords to native hyperbolic coords"""
    rs = np.linalg.norm(coords, axis=1)
    native_rs = 2.0 * np.arctanh(rs)
    return (coords.T * native_rs).T

def plot_poincare_disk(coords, labels=None, title="", ax=None, 
                        max_radius=None, show_circles=True):
    """Standard Poincaré disk visualization"""
    # Implementation from Fig2d.ipynb
```

---

### Example 1: Toggle Switch (Feature Matrix)

#### Cell 4: Toggle Switch - Introduction (Markdown)
- Dataset description
- Expected outcomes

#### Cell 5: Toggle Switch - Load Metadata (Python)
```python
# Load metadata for coloring
metadata = pd.read_csv('../data/metadata/ToggleSwitch_metadata.csv', index_col=0)
print(f"Dataset shape: {metadata.shape}")
print(f"Metadata columns: {metadata.columns.tolist()}")
```

#### Cell 6: Toggle Switch - Clustering Command (Markdown)
```
Run this command in terminal:
cd ../MultiscalehMDS_feature
python Generate_cls.py --featmat FeatMat_ToggleSwitch.txt --n-clusters 20
```

#### Cell 7: Toggle Switch - Embedding Command (Markdown)
```
Run this command in terminal:
python Multiscale_hmds.py --featmat FeatMat_ToggleSwitch.txt --clusters cls_20 \
       --neighbors 10 --dimension 2 --min-cluster-size 1 --max-cluster-size 400 \
       --save-matrix --compute-metrics --outlier-neighbors 10
```

#### Cell 8: Toggle Switch - Load Results (Python)
```python
# Load embedding coordinates
coords_file = '../MultiscalehMDS_feature/test/FeatMat_ToggleSwitch/cls_20_Nn_10_Nd_2_min_1_max_400_coords_*.txt'
# Handle two files (centroids and individuals)
# Load index file to reorder
# Convert to native coordinates
```

#### Cell 9: Toggle Switch - Visualize (Python)
```python
# Create Poincaré disk visualization
fig, ax = plt.subplots(figsize=(8, 8))
# Adapted from supp-visualization.ipynb but simpler
# Color by metadata categories
```

#### Cell 10: Toggle Switch - Quality Metrics (Python)
```python
# Load and display metrics
metrics = pd.read_csv('..path../metrics.csv')
print("Embedding quality metrics:")
display(metrics)
```

---

### Example 2: WordNet Mammals (Distance Matrix)

#### Cell 11: WordNet - Introduction (Markdown)
- Dataset description
- Hierarchical taxonomy explanation

#### Cell 12: WordNet - Clustering Command (Markdown)
```
Run this command in terminal:
cd ../MultiscalehMDS_distance
python Generate_cls.py --distmat DistMat_mammalwords.txt --thresholds 0.4
```

#### Cell 13: WordNet - Embedding Command (Markdown)
```
python Multiscale_hmds.py --distmat DistMat_mammalwords.txt --clusters cls_0.4 \
       --neighbors 60 --dimension 2 --min-cluster-size 1 --max-cluster-size 300 \
       --save-matrix --compute-metrics --outlier-neighbors 10
```

#### Cell 14: WordNet - Load Results (Python)
```python
# Load embedding coordinates
# Load node names/metadata
# Load curvature from metrics
# Convert to native coordinates
```

#### Cell 15: WordNet - Basic Visualization (Python)
```python
# Simple Poincaré disk plot
# Color by distance from root
# Size by node importance
```

#### Cell 16: WordNet - Enhanced Visualization (Python)
```python
# Adapted from Fig2d.ipynb last cell
# Poincaré coordinates with:
# - Hyperbolic circles
# - Labels for important nodes
# - Color by hierarchy level
# - Size by number of descendants
```

#### Cell 17: WordNet - Hierarchy Analysis (Python)
```python
# Compute hierarchy statistics
# Distance from origin vs. tree depth
# Visualization of hierarchy preservation
```

---

### General Guidelines

#### Cell 18: Parameter Selection Tips (Markdown)
- Quick reference table
- When to increase/decrease neighbors
- How to choose dimensions
- Cluster size considerations

#### Cell 19: Troubleshooting (Markdown)
- Common issues and solutions
- File not found errors
- Memory issues
- Quality metric interpretation

#### Cell 20: Next Steps (Markdown)
- Links to parameter guide
- Adapting code for custom data
- Performance optimization tips

---

## Implementation Checklist

### Phase 1: Markdown Document
- [ ] Write Introduction section
- [ ] Complete Toggle Switch example (sections 2.1-2.6)
- [ ] Complete WordNet example (sections 3.1-3.6)
- [ ] Write "Working with Your Own Data" section
- [ ] Add troubleshooting section
- [ ] Review and polish

### Phase 2: Jupyter Notebook
- [ ] Set up imports and helper functions
- [ ] Implement Toggle Switch example (cells 4-10)
- [ ] Implement WordNet example (cells 11-17)
- [ ] Add general guidelines (cells 18-20)
- [ ] Test all code cells
- [ ] Add figure placeholders
- [ ] Review outputs

### Phase 3: Integration
- [ ] Ensure consistency between markdown and notebook
- [ ] Cross-reference with Overview and Parameters docs
- [ ] Verify all file paths are correct
- [ ] Test commands in both directories
- [ ] Add "Note" boxes for common pitfalls

---

## Key Considerations

### File Paths
- **Original results**: `MultiscalehMDS_*/result/`
- **Tutorial results**: `MultiscalehMDS_*/test/`
- Notebook should look in `test/` by default
- Provide fallback to `result/` for demonstration

### Code Reusability
- Extract common plotting functions
- Make code easy to adapt for custom data
- Clear comments explaining each step

### Pedagogical Approach
- Start simple (Toggle Switch with basic viz)
- Progress to complex (WordNet with network structure)
- Explain *why* not just *how*
- Connect to theoretical concepts from Overview

### Visual Consistency
- Use consistent color schemes
- Standardize figure sizes
- Include figure captions
- Add placeholders for images in markdown

### Accessibility
- Explain hyperbolic geometry concepts briefly
- Provide intuition for parameters
- Include expected outputs (shapes, sizes)
- Warn about computational requirements

---

## Dependencies

### Python Packages
- numpy, pandas, matplotlib, seaborn
- textalloc (for WordNet labels)
- scipy (for metrics)
- Local: MHMDS.embed_funs

### Data Files Required
- `data/FeatMat_ToggleSwitch/FeatMat_ToggleSwitch.txt`
- `data/DistMat_mammalwords/DistMat_mammalwords.txt`
- `data/metadata/ToggleSwitch_metadata.csv`
- `data/metadata/mammal_metadata.csv` (or create from Fig2d.ipynb)

### Output Files (for testing)
- Should be generated fresh by tutorial commands
- Provide checksums or expected sizes for validation

---

## Success Criteria

### For Markdown
- [ ] Clear step-by-step instructions
- [ ] All commands are copy-pasteable
- [ ] Output files are clearly described
- [ ] Links to other docs work
- [ ] Troubleshooting covers common issues

### For Notebook
- [ ] All cells run without errors
- [ ] Visualizations are informative
- [ ] Code is well-commented
- [ ] Easy to adapt for custom data
- [ ] Execution time is reasonable (<5 min total)

---

## Timeline Estimate

- Markdown document: 3-4 hours
- Jupyter notebook: 2-3 hours
- Testing and refinement: 1-2 hours
- **Total**: 6-9 hours

---

## Notes for Implementation

1. **Toggle Switch**: Keep simple, focus on workflow
2. **WordNet**: Show off hyperbolic geometry features
3. **Balance**: Theory (markdown) vs. Practice (notebook)
4. **Flexibility**: Make code adaptable for readers' data
5. **Testing**: Verify all paths work from docs/ directory
