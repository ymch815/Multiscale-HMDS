# Optimization backend refactoring plan

## Goal

Replace CmdStanPy as the default MuH-MDS optimization backend with PyTorch
automatic differentiation and L-BFGS. Preserve the published mathematical
objectives, global-local algorithm, public functions, and output formats.

The exact paper implementation is preserved by the local `paper-v1.0` tag.
The `.stan` sources remain in the repository as scientific references and as
an optional regression backend.

## Work phases

1. Record the paper-code revision and establish deterministic baseline cases.
2. Implement shared, vectorized Lorentz geometry in double precision.
3. Translate `CM2.stan`, `relax.stan`, `lorentz2.stan`, and `transform.stan`.
4. Keep existing public functions and NumPy return values compatible.
5. Validate objectives, gradients, optimized distance matrices, embedding
   metrics, CLI outputs, runtime, and memory.
6. Remove CmdStan compilation from the default installation and document the
   optional legacy setup.
7. Test supported Python versions and NumPy 2 on Linux and macOS before the
   final release.

## Numerical policy

- Use `torch.float64`, CPU execution, explicit seeds, and finite-value checks.
- Clamp the Lorentz inner product to `1 + eps` before `acosh` to handle
  floating-point excursions outside its domain.
- Represent positive `lambda` and `sigma` through a smooth positive transform.
- Compare embeddings through loss and distance-based quality metrics, not raw
  coordinates, because equivalent embeddings may differ by an isometry.

## Merge criteria

- Core installation and execution require no CmdStan or C++ compiler.
- Objective values agree with the Stan definitions at identical parameters.
- Unit, integration, and small end-to-end tests pass.
- Existing command-line workflows and saved formats remain compatible.
- Documentation contains working installation and migration instructions.
- The repository owner reviews the local commits and performs the final push.

