# ApproxCholesky.jl

Julia bindings for the C API in [apxchol](https://github.com/AlgOptGroup/apxchol).
The package builds a shared library from a pinned apxchol revision and links it
to `LLVMOpenMP_jll`, the OpenMP runtime loaded by Julia. Using one runtime
avoids the macOS segfault caused by loading Julia's `libomp` and Homebrew
`libomp` into the same process.

This repository is private while the C API and binary distribution are under
review.

## Build

Install CMake and a C++23 compiler. On macOS, Xcode command line tools are
sufficient. Then run:

```julia
import Pkg
Pkg.instantiate()
Pkg.build("ApproxCholesky")
Pkg.test("ApproxCholesky")
```

The build clones the pinned apxchol revision. To build against a local
checkout instead:

```julia
ENV["APXCHOL_SOURCE_DIR"] = "/path/to/apxchol"
Pkg.build("ApproxCholesky")
```

## Use

```julia
using ApproxCholesky
using SparseArrays

A = sparse([2.0 -1.0; -1.0 2.0])
b = [1.0, 0.0]
result = solve(A, b; tolerance=1e-10, threads=4)

result.x
result.iterations
result.relative_residual
result.setup_seconds
result.solve_seconds
```

`A` must be a `SparseMatrixCSC` containing a Laplacian or SDDM matrix.
`threads=0` leaves the OpenMP runtime setting unchanged. Setting a positive
thread count changes the OpenMP setting for the calling process, so concurrent
calls with different thread counts are unsupported.

## Apple Silicon

Apple Silicon uses apxchol's CPU/OpenMP backend. CUDA requires an NVIDIA GPU
and is unavailable on macOS. The package does not contain a Metal backend.
A Metal implementation would need sparse triangular solve kernels, a device
PCG loop, and performance tests against the CPU path before it could be exposed
through this API.

## Distribution status

The current source build is intended for development and integration testing.
A registered package should replace it with platform artifacts built through
BinaryBuilder and an `apxchol_jll` dependency.
