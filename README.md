# ApxChol.jl

Julia bindings for [apxchol](https://github.com/AlgOptGroup/apxchol), a parallel
approximate-Cholesky preconditioned conjugate-gradient solver for graph
Laplacian, SDDM and lumpable M-matrix systems. The package builds `libapxchol`
from the apxchol revision in `deps/apxchol.pin` and calls it through apxchol's
C ABI (`include/apxchol/c_api.h`).

## Install

Linux or macOS, CMake ≥ 3.20 and a C++23 compiler (Apple Clang from Xcode 16,
GCC ≥ 14 or Clang ≥ 18). The build configures apxchol with its own CMake core
against `LLVMOpenMP_jll`'s OpenMP runtime; on Apple silicon it includes the
Metal backend.

```julia
import Pkg
Pkg.add(url="https://github.com/eigenergy/ApxChol.jl")   # builds libapxchol
Pkg.test("ApxChol")
```

`ENV["APXCHOL_SOURCE_DIR"]` builds a local apxchol checkout instead of the pin;
`CXX` selects the compiler. The tests include a C program compiled against the
same library whose factor, solution and residual must match Julia's byte for
byte.

## Use

```julia
using ApxChol, LinearAlgebra, SparseArrays

S = ApxCholSolver(A; rtol=1e-10)     # factorize once
x = S \ b                            # solve; throws NotConvergedError otherwise
ldiv!(x, S, b)                       # into x (x may be b)
X = S \ B                            # every column of B

r = ApxChol.solve!(x, S, b)          # never throws on non-convergence
r.converged, r.iterations, r.relative_residual, r.solve_seconds
```

`A` is a `SparseMatrixCSC` with real values and `Int32` or `Int64` indices, or
a `Symmetric`/`Hermitian` wrapper of one: a Laplacian, an SDDM matrix, or an
M-matrix with a few positive off-diagonals (lumped onto the diagonal for the
preconditioner only). Anything else throws `ApxCholError(:invalid_operator)`.
Laplacian systems return the min-norm solution and need a right-hand side
compatible with every connected component.

`relative_residual` is ‖b − A x‖/‖b‖ of the returned `x`, recomputed in FP64
from the original operator on every backend; `converged` means it is below
`rtol`. Contiguous `Float64` vectors and views are passed to apxchol in place,
and a warmed-up `solve!`, `ldiv!` or `apply!` allocates nothing. apxchol
rescales extreme right-hand sides internally; a `b` and `x0` that cannot share
one scale throw `ApxCholError(:invalid_argument)`, and a solution outside the
finite range throws `ApxCholError(:unsupported)`.

## Options

Keywords of `ApxCholSolver` and of the one-shot `ApxChol.solve(A, b; ...)`.
Unset keywords take apxchol's defaults, which `ApxChol.options(S)` reports.

| keyword | default | meaning |
|---|---|---|
| `backend` | `:auto` | CPU for ≤ 32,768 rows, when adopting a factor, and for inputs or options Metal cannot take; otherwise Metal when usable. Also `:cpu` or `:metal` |
| `rtol` | `1e-8` | relative residual target |
| `maxiter` | `200` | PCG iteration limit |
| `stagnation_window` | `50` | stop when the residual has not halved in this many iterations; 0 disables |
| `factor_storage` | `nothing` | triangular-solve storage, `Float16` or `Float32`; `nothing` reads `APXCHOL_FACTOR_STORAGE`, then the backend default (`Float32`) |
| `threads` | `0` | ceiling on apxchol's OpenMP workers per call; 0 inherits the caller's limit |
| `keep_factor` | `false` | retain the factor values for `ApxChol.factor` and `ApxCholFactor(S)` |
| `seed` | `42` | sampling seed |
| `sampler` | `:gks` | or `:trace_cycle` |
| `partitioner` | `:block_greedy` | or `:priority_greedy`, `:baumann_kyng` |
| `degree_quantile` | by route | independent-set degree cap |
| `exact_clique_max_degree` | `0` | exact cliques up to this degree |
| `omp_threshold` | `2000` | active vertices before the factorization uses OpenMP |

`ApxChol.solve`/`solve!` also take `rtol`, `maxiter` and `x0` (`X0` for
matrices) per call. `ApxChol.stats(S)` reports the route chosen, the installed
storage, the factor size, rounds, lumped entries and setup time.

## Factors

```julia
F = ApxCholFactor(A; seed=7)         # factorize on the CPU, without a solver
S1 = ApxCholSolver(A, F)             # A preconditioned by F
S2 = ApxCholSolver(A2, F)            # another operator of the same size
F2 = ApxCholFactor(S)                # the factor of a solver made with keep_factor=true

f = ApxChol.factor(F)                # f.L::SparseMatrixCSC{Float32,Int64}, f.p; A[f.p, f.p] ≈ f.L * f.L'
ApxChol.ordering(S)                  # f.p alone, always available
```

Each solver adopts its own copy, so factors and solvers close in any order. A
Laplacian factor refuses an SDDM operator; an SDDM factor may precondition a
Laplacian, whose solution is then not centred.

## Preconditioner

```julia
using Krylov
P = ApxChol.aspreconditioner(S)      # ldiv!(y, P, x) applies M⁻¹ once, allocation-free
x, stats = Krylov.cg(A, b; M=P, ldiv=true)
```

With an SDDM factor `P` is a fixed linear map. With a Laplacian factor it is
fixed up to a constant vector: the CPU route centres every tenth application,
Metal every one.

## Metal

On Apple silicon, `ApxChol.backend_available(:metal)` reports whether the GPU
is usable. A Metal solver builds its factor on the GPU, so it differs from the
CPU's for the same seed; `ApxCholSolver(A, F; backend=:metal)` adopts a CPU
factor instead. Solves run one right-hand side at a time with double-float
Krylov vectors, and the residual is recomputed in FP64. Options the Metal setup
cannot take (such as `sampler=:trace_cycle`) throw `ApxCholError(:unsupported)`
under `backend=:metal` and fall back to the CPU under `:auto`.

## Safety

A solver or factor owns native memory, released by `close` or its finalizer;
calls after `close` throw `InvalidStateException`. Every native call keeps its
handle and arrays rooted and checks array lengths first. Calls on one solver
are serialized by its lock; distinct solvers run concurrently from Julia
threads, and on Julia ≥ 1.12 native calls are GC-safe. The library links only
`LLVMOpenMP_jll`'s `libomp`; on load the package sets `KMP_GTID_MODE=2` unless
already set, and checks that the library matches the pin, the `libomp` and
this package's C ABI layout.

## Limits

`n` and `nnz(A)` are at most 2³¹ − 1; factors beyond 2³² − 1 entries need an
apxchol build with `APXCHOL_64BIT_EDGE_INDICES=ON`. CUDA is not exposed by the
C ABI. Do not mutate `A` during a call.

## License

MIT for the Julia wrapper. The native library keeps apxchol's BSD-4-Clause
license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This product
includes software developed by ETH Zürich and the apxchol contributors.
