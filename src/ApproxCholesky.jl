module ApproxCholesky

using LLVMOpenMP_jll
using Libdl
using LinearAlgebra
using SparseArrays

const library_name = Sys.iswindows() ? "approxcholesky.dll" :
                     Sys.isapple() ? "libapproxcholesky.dylib" : "libapproxcholesky.so"
const library_directory = Sys.iswindows() ? "bin" : "lib"
const libapproxcholesky = normpath(
    joinpath(@__DIR__, "..", "deps", "usr", library_directory, library_name))

export SolveResult, solve

struct CSolveInfo
    iterations::Cint
    relative_residual::Cdouble
    setup_seconds::Cdouble
    solve_seconds::Cdouble
end

struct SolveResult
    x::Vector{Float64}
    iterations::Int
    relative_residual::Float64
    setup_seconds::Float64
    solve_seconds::Float64
    converged::Bool
end

function __init__()
    !isfile(libapproxcholesky) &&
        error("ApproxCholesky is not built; run `import Pkg; Pkg.build(\"ApproxCholesky\")`")
    # Load the same runtime used to link the C++ library before loading the
    # library itself. Loading a second OpenMP runtime in Julia can segfault once
    # the solver enters a parallel region.
    Libdl.dlopen(LLVMOpenMP_jll.libomp_path, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    Libdl.dlopen(libapproxcholesky, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
end

function error_message(buffer::Vector{UInt8})
    terminator = something(findfirst(iszero, buffer), length(buffer) + 1)
    return String(buffer[1:(terminator - 1)])
end

"""
    solve(A, b; tolerance=1e-8, max_iterations=1000, seed=1, threads=0)

Solve a sparse Laplacian or SDDM system with apxchol. `threads=0` leaves the
OpenMP runtime setting unchanged; a positive value selects the number of
threads for this call.
"""
function solve(
    A::SparseMatrixCSC{<:Real,<:Integer},
    b::AbstractVector{<:Real};
    tolerance::Real=1e-8,
    max_iterations::Integer=1000,
    seed::Integer=1,
    threads::Integer=0,
)
    n, m = size(A)
    n == m || throw(DimensionMismatch("A must be square"))
    length(b) == n || throw(DimensionMismatch("b must have length $n"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    max_iterations <= typemax(Cint) || throw(ArgumentError("max_iterations exceeds Cint"))
    0 <= seed <= typemax(UInt32) || throw(ArgumentError("seed must fit in UInt32"))
    threads >= 0 || throw(ArgumentError("threads must be nonnegative"))
    threads <= typemax(Cint) || throw(ArgumentError("threads exceeds Cint"))
    n <= typemax(Cint) || throw(ArgumentError("A exceeds the current C++ index range"))
    nnz(A) <= typemax(Cint) || throw(ArgumentError("A exceeds the current C++ index range"))

    matrix = SparseMatrixCSC{Float64,Int64}(A)
    rhs = Vector{Float64}(b)
    solution = Vector{Float64}(undef, n)
    info = Ref(CSolveInfo(0, 0.0, 0.0, 0.0))
    error_buffer = zeros(UInt8, 512)

    status = ccall(
        (:apxchol_solve_csc64, libapproxcholesky),
        Cint,
        (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Cint,
         Ptr{Cdouble}, Cdouble, Cint, UInt32, Cint, Ptr{Cdouble},
         Ref{CSolveInfo}, Ptr{UInt8}, Int64),
        n,
        nnz(matrix),
        matrix.colptr,
        matrix.rowval,
        matrix.nzval,
        1,
        rhs,
        tolerance,
        max_iterations,
        seed,
        threads,
        solution,
        info,
        error_buffer,
        length(error_buffer),
    )
    status == 0 || error("apxchol failed: $(error_message(error_buffer))")

    stats = info[]
    return SolveResult(
        solution,
        stats.iterations,
        stats.relative_residual,
        stats.setup_seconds,
        stats.solve_seconds,
        stats.relative_residual <= tolerance,
    )
end

end
