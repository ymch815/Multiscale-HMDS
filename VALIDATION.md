# PyTorch backend validation

This document records checks performed while replacing CmdStanPy with the
PyTorch optimization backend. Results below were obtained on Apple Silicon
with Python 3.11, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, pandas 3.0.6,
and PyTorch 2.14.0.

## Objective convention

CmdStan optimization evaluates log densities proportionally and omits terms
that are constant with respect to all parameters. The PyTorch objectives use
the same convention. This does not affect the optimizer and lets objective
values be compared directly with CmdStan's `lp__` after changing sign.

## Unit tests

The test suite covers:

- Lorentz distance symmetry and a known radial distance.
- Local relaxation convergence.
- Global output shapes, positivity, and finite values.
- A term-by-term check of the global Stan proportional target.
- An automatic-gradient check against central finite differences.
- Center-of-mass symmetry.
- Outlier transformation and positive uncertainty.

Run it with:

```bash
python -m pytest -q
```

## Paper executable comparison

The repository's original `lorentz2` executable was run on the 20 centroids
of the Toggle Switch dataset with 1,000 L-BFGS iterations. At its returned
parameters, the PyTorch log target was `488.82628695`, versus the Stan CSV
`lp__` value `488.826`. The residual difference of `2.87e-4` is consistent
with the Stan CSV's default output precision. Both targets exclude the same
parameter-independent normalizing constants. This checks the likelihood,
inverse-gamma priors, curvature prior, Lorentz coordinate layout, and Stan CSV
array ordering together.

Optimizer endpoints need not match exactly: the objective is non-convex, the
two L-BFGS implementations use different histories and line searches, and an
embedding is non-identifiable up to hyperbolic isometry.

## End-to-end smoke test

The existing public API was run on all 200 samples in
`FeatMat_ToggleSwitch`, using 20 clusters, 10 neighboring centroids, and a
three-dimensional embedding. The run completed with:

```text
coordinate shape:       (200, 3)
all coordinates finite: true
distance correlation:   0.9999999781
```

The distance correlation compares upper-triangular input Euclidean distances
with output hyperbolic distances. This smoke test exercises global embedding,
classical-MDS initialization, every local relaxation, coordinate conversion,
and final integration.

## Remaining release checks

GitHub Actions is configured to run the unit tests on Linux and macOS with
Python 3.9 through 3.13. Those remote jobs will run after the repository owner
pushes the branch. Larger biological datasets and paper-figure regeneration
remain appropriate pre-release checks but are not required for every commit.
