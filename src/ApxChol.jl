"""
    ApxChol

Julia bindings for [apxchol](https://github.com/AlgOptGroup/apxchol), a parallel
approximate-Cholesky preconditioned conjugate-gradient solver for graph
Laplacian, SDDM and lumpable M-matrix systems, through a CxxWrap module over
apxchol's C++ solver.

Factor once with [`ApxCholSolver`](@ref), then solve any number of right-hand
sides with `S \\ b`, `ldiv!` or [`ApxChol.solve!`](@ref);
[`ApxChol.aspreconditioner`](@ref) exposes one preconditioner application for
Krylov solvers. An [`ApxCholFactor`](@ref) carries one factorization to any
number of solvers, including solvers for another operator of the same size.
"""
module ApxChol

using CxxWrap: CxxWrap, CxxPtr
using LLVMOpenMP_jll: LLVMOpenMP_jll
using Libdl: Libdl
using LinearAlgebra
using SparseArrays

export ApxCholSolver, ApxCholFactor, ApxCholError, NotConvergedError

include("native.jl")
include("handles.jl")
include("solve.jl")

@static if VERSION >= v"1.11"
    eval(Expr(:public, :solve, :solve!, :apply!, :aspreconditioner, :factor, :ordering, :stats,
              :options, :version, :build_info, :backend_available, :max_threads,
              :openmp_enabled, :SolveResult, :BlockSolveResult, :ExportedFactor,
              :ApxCholPreconditioner))
end

"""
    ApxChol.version() -> String

The apxchol version and git revision the library was built from.
"""
version() = String(Native.version())

"""
    ApxChol.openmp_enabled() -> Bool

Always `true`: apxchol requires OpenMP.
"""
openmp_enabled() = true

"""
    ApxChol.max_threads() -> Int

The calling thread's OpenMP team limit; solvers created with `threads=0` use it.
"""
max_threads() = Int(Native.max_threads())

"""
    ApxChol.backend_available(backend::Symbol) -> Bool

Whether this build and machine support `backend`: `:cpu`, `:auto`, or `:metal`
(Apple silicon builds with a usable GPU).
"""
backend_available(backend::Symbol) =
    haskey(BACKENDS, backend) && Bool(Native.backend_available(code(BACKENDS, backend, "backend")))

function __init__()
    DEFAULTS[] = Options(Native.default_options()...)
    return nothing
end

end
