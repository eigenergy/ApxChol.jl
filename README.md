# ApxChol.jl

Julia bindings for [apxchol](https://github.com/AlgOptGroup/apxchol), a parallel
approximate-Cholesky preconditioned conjugate-gradient solver for graph
Laplacian, SDDM and lumpable M-matrix systems. The package builds `libapxchol`
from a pinned apxchol revision and calls it only through apxchol's C ABI
(`include/apxchol/c_api.h`), which it mirrors function for function.

The package was previously named ApproxCholesky.jl; its UUID is unchanged.

## Build

Requirements: Linux or macOS, CMake ≥ 3.20, and a C++23 compiler (Apple Clang
from Xcode 16 or newer, GCC ≥ 14, or Clang ≥ 18). The build fetches the pinned
apxchol revision and configures it with apxchol's own `cmake/apxchol_core.cmake`,
as apxchol's Python bindings do, against `LLVMOpenMP_jll`'s OpenMP runtime.

```julia
import Pkg
Pkg.build("ApxChol")
Pkg.test("ApxChol")
```

`ENV["APXCHOL_SOURCE_DIR"] = "/path/to/apxchol"` builds a local checkout instead
of the pin; `CXX` selects the compiler. The library is tuned for the build
machine. On Apple silicon it includes apxchol's Metal backend.

## Quick start

```julia
using ApxChol, LinearAlgebra, SparseArrays

S = ApxCholSolver(A; rtol=1e-10)     # factorize once
x = S \ b                            # solve; throws NotConvergedError if it does not converge
ldiv!(x, S, b)                       # into x (x may be b itself)
X = S \ B                            # every column of B

r = ApxChol.solve!(x, S, b)          # never throws on non-convergence:
r.converged, r.iterations, r.relative_residual, r.solve_seconds
```

`A` is a `SparseMatrixCSC` with real values and `Int32` or `Int64` indices, or a
`Symmetric`/`Hermitian` wrapper of one. It must be a Laplacian, an SDDM matrix
or a nearly-SDDM M-matrix: symmetric, positive diagonal, nonpositive
off-diagonals. apxchol validates it and never reinterprets it; a few positive
off-diagonals are lumped onto the diagonal for the preconditioner only, and an
adjacency matrix is rejected with `ApxCholError(:invalid_operator)`. Laplacian
systems return the min-norm solution and need a right-hand side compatible with
every connected component.

`b`, `x` and `x0` may be any real vectors. Contiguous `Float64` vectors, views
included, are handed to apxchol in place, and a warmed-up `solve!`, `ldiv!` or
`apply!` on them allocates nothing in Julia. Other vectors are converted once.

## Convergence and the reported residual

`relative_residual` is ‖b − A x‖/‖b‖ of the returned `x`, recomputed by apxchol
in FP64 from the original operator on every backend; `converged` means it is
below `rtol`. Right-hand sides of any scale are solved: apxchol rescales an
extreme `b` and `x0` by one exact power of two internally. A `b` and `x0` that
cannot share a scale throw `ApxCholError(:invalid_argument)`; a solution outside
the finite range throws `ApxCholError(:unsupported)` and leaves `x` unspecified.
`solve` and `solve!` report non-convergence in their result; `\` and `ldiv!`
have no flag to return and throw `NotConvergedError(result)` instead. Otherwise
the returned iterate is always written.

## Factor reuse and export

```julia
F = ApxCholFactor(A; seed=7)         # factorize once, without a solver
S1 = ApxCholSolver(A, F)             # A preconditioned by F
S2 = ApxCholSolver(A2, F)            # another operator of the same size
F2 = ApxCholFactor(S)                # the factor of a solver made with keep_factor=true

f = ApxChol.factor(F)                # ExportedFactor
f.L, f.p                             # A[f.p, f.p] ≈ f.L * f.L'  (L::SparseMatrixCSC{Float32,Int64})
ApxChol.ordering(S)                  # f.p, always available
```

Each solver adopts its own copy of the factor, so factors and solvers close in
any order. `ApxCholSolver(A2, F)` takes only the solve keywords; `A2` must have
`F`'s size. A Laplacian factor refuses an SDDM operator; an SDDM factor may
precondition a Laplacian, whose solution is then not centred. On the CPU route,
`ApxCholSolver(A, ApxCholFactor(A))` is bit for bit the solver `ApxCholSolver(A)`
builds. A Metal solver builds its own factor on the GPU (see below), so there
only solvers adopting the same factor agree bit for bit.

## Options

Keywords of `ApxCholSolver` (and of the one-shot `ApxChol.solve(A, b; ...)`);
unset keywords take apxchol's defaults, which `ApxChol.options(S)` reports.

| keyword | default | meaning |
|---|---|---|
| `backend` | `:auto` | `:auto` chooses as apxchol's C++ solver does (CPU for ≤ 32,768 rows, when adopting a factor, and for operators or options Metal cannot take; else Metal when usable); `:cpu`; `:metal` |
| `rtol` | `1e-8` | relative residual target |
| `maxiter` | `200` | PCG iteration limit |
| `stagnation_window` | `50` | stop if the residual has not halved in this many iterations (0 disables) |
| `factor_storage` | `nothing` | triangular-solve storage `Float16` or `Float32`; `nothing` reads `APXCHOL_FACTOR_STORAGE`, then the backend default (`Float32` on CPU and Metal) |
| `threads` | `0` | ceiling on apxchol's OpenMP workers in every call on the handle; 0 inherits the calling thread's limit |
| `keep_factor` | `false` | retain the factor values for `ApxChol.factor` and `ApxCholFactor(S)` |
| `seed` | `42` | sampling seed |
| `sampler` | `:gks` | also `:trace_cycle` |
| `partitioner` | `:block_greedy` | also `:priority_greedy`, `:baumann_kyng` |
| `degree_quantile` | by route | independent-set degree cap (apxchol's `factor_options.h`) |
| `exact_clique_max_degree` | `0` | exact cliques up to this degree |
| `omp_threshold` | `2000` | active vertices before the factorization engages OpenMP |

`ApxChol.solve!`/`ApxChol.solve` also take `rtol`, `maxiter` and `x0` (`X0` for
matrices) per call. `ApxChol.stats(S)` reports the route chosen (`backend`), the
installed `factor_storage`, the factor size, rounds, lumped entries and setup
time.

## Preconditioner for other Krylov solvers

```julia
using Krylov
P = ApxChol.aspreconditioner(S)      # ldiv!(y, P, x) applies M⁻¹ once, allocation-free
x, stats = Krylov.cg(A, b; M=P, ldiv=true)
```

With an SDDM factor `P` is a fixed linear map. With a Laplacian factor it is
fixed only up to a constant vector: the CPU route centres every 10th
application (apxchol's center-k schedule, `APXCHOL_CENTER_K`), Metal every one.

## Safety and concurrency

- **Memory.** A solver or factor owns native memory, released by `close` or its
  finalizer. Every native call keeps its handle and arrays rooted, so a
  finalizer can never run under it; after `close` every call throws
  `InvalidStateException`. Array lengths are checked in Julia before any
  pointer reaches apxchol, which checks every index it reads. Partially
  overlapping input and output arrays are copied; exact aliases are allowed.
- **Threads.** Calls on one solver are serialized by its lock, and `close`
  waits for a call in flight. Reads of one factor (adoptions, `stats`,
  `factor`) run concurrently, and `close` waits for them. Distinct solvers run
  concurrently from different Julia threads; give each a small `threads`.
- **Garbage collection.** On Julia 1.12 and later native calls are GC-safe:
  other threads keep collecting garbage while a long factorization or solve runs.
- **One OpenMP runtime.** The library links `LLVMOpenMP_jll`'s `libomp`, which
  Julia packages share. On load the package sets `KMP_GTID_MODE=2` unless you
  have, so concurrent solves on Julia's task stacks get distinct OpenMP thread
  identities.
- **Load-time check.** `using ApxChol` verifies that the library is the pinned
  revision, built against the current `libomp`, with this package's C ABI
  layout; otherwise it asks for `Pkg.build("ApxChol")`.

## Parity with the C ABI

| C ABI | ApxChol |
|---|---|
| `apxchol_solver_create` | `ApxCholSolver(A; ...)` |
| `apxchol_solver_create_from_factor` | `ApxCholSolver(A, F; ...)` |
| `apxchol_solver_solve` | `S \ b`, `ldiv!`, `ApxChol.solve!`, `ApxChol.solve` |
| `apxchol_solver_solve_block` | the same with matrices |
| `apxchol_solver_apply` | `ApxChol.apply!`, `ApxChol.aspreconditioner` |
| `apxchol_solver_stats`, `apxchol_factor_stats` | `ApxChol.stats` |
| `apxchol_solver_export_factor`, `apxchol_factor_export` | `ApxChol.factor`, `ApxChol.ordering` |
| `apxchol_factor_create` | `ApxCholFactor(A; ...)` |
| `apxchol_solver_copy_factor` | `ApxCholFactor(S)` |
| `apxchol_solver_destroy`, `apxchol_factor_destroy` | `close`, finalizers |
| `apxchol_options_default` | the keyword defaults (`ApxChol.options`) |
| `apxchol_backend_available`, `apxchol_get_max_threads`, `apxchol_openmp_enabled`, `apxchol_version` | `ApxChol.backend_available`, `max_threads`, `openmp_enabled`, `version` |
| `apxchol_c_abi_version` | checked when the package loads |

The tests compile a C program against the same library, solve the same system
through both at one thread, and require identical factor, ordering, solution and
residual bytes (`test/digest.jl`).

## Limits

- `n` and `nnz(A)` must each be at most 2³¹ − 1; do not mutate `A` during a call.
- Factors with more than 2³² − 1 entries abort the process; use an apxchol build
  with `APXCHOL_64BIT_EDGE_INDICES=ON` for those. An exception escaping a native
  OpenMP region terminates the process.
- CUDA is not exposed by the C ABI.

## Apple silicon and Metal

`ApxChol.backend_available(:metal)` reports whether the GPU is usable. A Metal
solver builds its factor on the GPU (apxchol's block-greedy GKS setup, degree
quantile 0.2 by default), so its factor and ordering differ from the CPU
route's for the same seed. `ApxCholFactor` always factorizes on the CPU, and
`ApxCholSolver(A, F; backend=:metal)` adopts that host factor. Solves run one
right-hand side at a time on the GPU with double-float Krylov vectors (about
48 bits); `relative_residual` is recomputed in FP64. `factor_storage=Float16`
stores the triangular-solve factor values in FP16. Options the Metal setup
cannot take (for example `sampler=:trace_cycle`) throw
`ApxCholError(:unsupported)` under `backend=:metal`; `:auto` uses the CPU for
them. See apxchol's `docs/precision.md` for the precision contract.

## Migrating from 0.3

- `ApxChol.factor` returns an `ExportedFactor` with `L::SparseMatrixCSC{Float32,Int64}`
  (apxchol's native values) and `p`, the LinearAlgebra convention
  (`A[p, p] ≈ L L'`). 0.3 returned `(L, perm)` with `perm = invperm(p)`;
  destructuring the new type throws. `ApxChol.permutation` is now `ApxChol.ordering`
  and returns `p`.
- `solve` and `solve!` are no longer exported: use `S \ b` and `ldiv!`, or
  `ApxChol.solve!`, or `using ApxChol: solve!`.
- `backend` defaults to `:auto`: on Apple silicon, inputs of more than 32,768
  rows now solve on the GPU. Pass `backend=:cpu` to keep the CPU route.
- The `storage` keyword is gone (apxchol has one graph layout); `factor_storage` is new.
- Closed handles throw `InvalidStateException` instead of `ArgumentError`.

## Pinned revision

`deps/apxchol.pin` names the immutable apxchol commit the package builds:
currently the head of [AlgOptGroup/apxchol#3](https://github.com/AlgOptGroup/apxchol/pull/3)
on [samtalki/apxchol](https://github.com/samtalki/apxchol). It moves to
`AlgOptGroup/apxchol` once that pull request merges; no branch name determines
the build.

## Distribution status

The source build is for development and integration testing. A registered
package should replace it with platform artifacts built through BinaryBuilder
and an `apxchol_jll` dependency; the library handle is one constant in
`src/capi.jl`.

## License

The Julia wrapper is MIT licensed. The native library retains apxchol's
BSD-4-Clause license; see [third-party notices](THIRD_PARTY_NOTICES.md).

This product includes software developed by ETH Zürich and the apxchol
contributors.
