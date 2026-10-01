# ApxChol.jl

Julia bindings for [apxchol](https://github.com/AlgOptGroup/apxchol), a parallel
approximate-Cholesky preconditioned conjugate-gradient solver for graph
Laplacian, SDDM and lumpable M-matrix systems. The package builds a shared
library, `libapxchol`, from a pinned apxchol revision and calls it through
apxchol's exception-safe C ABI.

This repository is private while the C ABI and binary distribution are under
review. (It was previously named ApproxCholesky.jl; the package UUID is unchanged.)

## Build

Requirements: Linux or macOS, CMake ≥ 3.20, and a C++23 compiler (Apple
Clang from Xcode 16 or newer, GCC ≥ 14, or Clang ≥ 18). The build downloads
the pinned apxchol revision and Eigen 3.4.0.

```julia
import Pkg
Pkg.instantiate()
Pkg.build("ApxChol")
Pkg.test("ApxChol")
```

To build against a local apxchol checkout instead of the pin:

```julia
ENV["APXCHOL_SOURCE_DIR"] = "/path/to/apxchol"
Pkg.build("ApxChol")
```

`CXX` selects the compiler. The library is tuned for the build machine
(`-mcpu=native` on Apple silicon).

## Use

```julia
using ApxChol, SparseArrays

S = ApxCholSolver(A; rtol=1e-10, threads=4)  # factorize once
r = solve(S, b)                              # SolveResult
r.x, r.iterations, r.relative_residual, r.converged

R = solve(S, B)                              # every column of a matrix B
x = S \ b                                    # throws NotConvergedError if not converged
solve(A, b; rtol=1e-10)                      # one-shot: factorize, solve, release
```

`A` must be a `SparseMatrixCSC` holding a Laplacian, SDDM matrix or nearly-SDDM
M-matrix: symmetric, positive diagonal, non-positive off-diagonals. apxchol
validates it and never reinterprets it. A matrix with a few positive
off-diagonals is accepted by lumping them onto the diagonal for the
preconditioner only (the solve still applies `A`). An adjacency matrix is
rejected with an `ApxCholError(:invalid_operator)` that says so.

Laplacian systems return the min-norm solution (`sum(x) ≈ 0`). The
right-hand side must be compatible with every connected component; it is not
projected.

### Convergence

`solve` and `solve!` never throw on non-convergence: check `r.converged`
(`relative_residual < rtol`). The returned iterate is always written. `\` and
`ldiv!` cannot return a flag, so they throw `NotConvergedError(result)` instead.

### Options

Keyword arguments of `ApxCholSolver` (and of the one-shot `solve(A, b; ...)`).
`nothing` selects the library default.

| keyword | default | meaning |
|---|---|---|
| `rtol` | `1e-8` | relative residual target ‖b − Ax‖/‖b‖ |
| `maxiter` | `200` | PCG iteration limit |
| `stagnation_window` | `50` | stop if the residual has not halved in this many iterations (0 disables) |
| `seed` | `42` | sampling seed |
| `threads` | `0` | OpenMP threads for every call on the solver; 0 inherits |
| `sampler` | `:gks` | clique sampler: `:gks` or `:trace_cycle` |
| `partitioner` | `:block_greedy` | also `:priority_greedy`, `:baumann_kyng` |
| `storage` | `:vec_pool_aos` | also `:vec`, `:bstr` |
| `degree_quantile` | by route | independent-set degree cap (see apxchol's `factor_options.h`) |
| `exact_clique_max_degree` | `0` | exact cliques up to this degree |
| `keep_factor` | `false` | retain the factor for `ApxChol.factor` |
| `backend` | `:cpu` | `:metal`: Apple-GPU block PCG (Apple silicon) |

`solve`, `solve!` and the block forms also take `rtol`, `maxiter` and
`x0`/`X0` per call.

### Preconditioner for other Krylov solvers

```julia
using Krylov
P = ApxChol.aspreconditioner(S)        # ldiv!(y, P, x) applies M⁻¹ once
x, stats = Krylov.cg(A, b; M=P, ldiv=true)
```

### Introspection

`ApxChol.stats(S)` reports the factor size, lumped entries, rounds, setup
time and whether the operator is SDDM. `ApxChol.factor(S)` returns `(L, perm)`
with `A[p, p] ≈ L * L'` for `p = invperm(perm)` (requires `keep_factor=true`).
`ApxChol.permutation(S)`, `ApxChol.options(S)` and `ApxChol.version()` are
also available.

## Threads and concurrency

- `threads = 0` uses the calling thread's OpenMP setting
  (`OMP_NUM_THREADS`, or `ApxChol.max_threads()`). A positive value applies to
  each call on that solver and restores the previous setting afterwards.
- Calls on one `ApxCholSolver` are serialized by a lock. Distinct solvers can
  be used concurrently from different Julia threads; give each a small
  `threads` to avoid oversubscribing the machine.
- The library links `LLVMOpenMP_jll`'s `libomp`, the runtime Julia packages
  share. One runtime per process avoids the crashes caused by mixing OpenMP
  runtimes.
- On load the package sets `KMP_GTID_MODE=2` (keyed thread-local lookup), unless you set it yourself.
  Otherwise, while few threads exist, libomp identifies a thread by its stack
  address. Julia tasks run on their own stacks, so concurrent solves on
  several Julia threads crashed on Julia 1.10. Load ApxChol before anything
  else starts OpenMP work.
- A solver holds native memory, released by its finalizer or by
  `close(S)`.

## Limits

- `n` and `nnz(A)` must each be at most 2³¹ − 1.
- Factors larger than 2³² − 1 entries abort the process. Use an apxchol build
  with `APXCHOL_64BIT_EDGE_INDICES=ON` for those.

## Pinned revision

`deps/apxchol.pin` names the apxchol revision the package builds. It
currently points at the `apple-silicon` branch of
[samtalki/apxchol](https://github.com/samtalki/apxchol), which proposes the
C ABI and the Metal backend upstream. It moves to `AlgOptGroup/apxchol` once
that branch is merged.

## Apple silicon and Metal

On Apple silicon the library is also built with apxchol's Metal backend.
That needs only the macOS SDK from the Command Line Tools: the kernels are
compiled at run time, so no Xcode Metal toolchain is required.
`ApxChol.backend_available(:metal)` reports whether the GPU is usable.

```julia
G = ApxCholSolver(A; backend=:metal, rtol=1e-10)
R = solve(G, B)      # up to 64 right-hand sides advance in lockstep per batch
```

- **Factor.** The factor is computed on the CPU exactly as for `:cpu`, and
  the GPU applies the same fp32 factor.
- **Precision.** The Krylov vectors (x, r, A p) and an inexact operator are
  double-float (about 48 bits), since Apple GPUs have no fp64.
- **Reported residual.** `relative_residual` is recomputed in fp64 against
  `A`, and `converged` means it is below `rtol`.
- **Reproducibility.** A column's result is independent of the other
  columns in the block.

See apxchol's `docs/precision.md` for the precision contract. Performance
relative to the CPU path has not been established. CUDA is unavailable on
macOS.

## Distribution status

The source build is intended for development and integration testing. A
registered package should replace it with platform artifacts built through
BinaryBuilder and an `apxchol_jll` dependency.
