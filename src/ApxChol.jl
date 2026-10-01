"""
    ApxChol

Julia bindings for [apxchol](https://github.com/AlgOptGroup/apxchol), a parallel
approximate-Cholesky preconditioned conjugate-gradient solver for graph
Laplacian, SDDM and lumpable M-matrix systems, through apxchol's C ABI.

Factor once with [`ApxCholSolver`](@ref), then [`solve`](@ref) any number of
right-hand sides; [`aspreconditioner`](@ref) exposes one preconditioner
application for external Krylov solvers.
"""
module ApxChol

using LLVMOpenMP_jll
using Libdl
using LinearAlgebra
using SparseArrays

export ApxCholSolver, SolveResult, BlockSolveResult, ApxCholError, NotConvergedError,
       solve, solve!

const libapxchol = normpath(joinpath(@__DIR__, "..", "deps", "usr", "lib",
                                     Sys.isapple() ? "libapxchol.dylib" : "libapxchol.so"))
const build_info_path = normpath(joinpath(@__DIR__, "..", "deps", "usr", "build_info.txt"))
const C_ABI_VERSION = 1

# ── C structs (include/apxchol/c_api.h, ABI version 1) ───────────────────────

struct COptions
    struct_size::UInt32
    backend::Int32
    tol::Float64
    max_iter::Int32
    stagnation_window::Int32
    seed::UInt32
    threads::Int32
    sampler::Int32
    partitioner::Int32
    storage::Int32
    keep_factor_values::Int32
    degree_quantile::Float64
    exact_clique_max_degree::UInt64
end

struct CSolveInfo
    struct_size::UInt32
    converged::Int32
    iterations::Int64
    relative_residual::Float64
    solve_seconds::Float64
end
CSolveInfo() = CSolveInfo(UInt32(sizeof(CSolveInfo)), 0, 0, 0.0, 0.0)

struct CStats
    struct_size::UInt32
    backend::Int32
    n::Int64
    factor_nnz::Int64
    lumped_offdiag::Int64
    rounds::Int64
    peak_graph_bytes::Int64
    setup_seconds::Float64
    sddm::Int32
    setup_max_threads::Int32
end
CStats() = CStats(UInt32(sizeof(CStats)), 0, 0, 0, 0, 0, 0, 0.0, 0, 0)

const STATUS = (:success, :not_converged, :invalid_argument, :invalid_operator,
                :no_factor_values, :unsupported, :out_of_memory, :internal_error)
const BACKENDS = (:cpu, :metal)
const SAMPLERS = (:gks, :trace_cycle)
const PARTITIONERS = (:block_greedy, :priority_greedy, :baumann_kyng)
const STORAGES = Dict(:vec => Int32(0), :bstr => Int32(2), :vec_pool_aos => Int32(4))

status_symbol(code::Integer) = 0 <= code < length(STATUS) ? STATUS[code + 1] : :unknown

function enum_code(table::Tuple, value::Symbol, what::String)
    i = findfirst(==(value), table)
    i === nothing && throw(ArgumentError("unknown $what :$value; valid: $(join(table, ", "))"))
    return Int32(i - 1)
end
function enum_code(table::AbstractDict, value::Symbol, what::String)
    haskey(table, value) ||
        throw(ArgumentError("unknown $what :$value; valid: $(join(sort!(collect(keys(table))), ", "))"))
    return table[value]
end
enum_symbol(table::Tuple, code::Integer) = table[code + 1]
enum_symbol(table::AbstractDict, code::Integer) = only(k for (k, v) in table if v == code)

# ── Errors and results ───────────────────────────────────────────────────────

"""
    ApxCholError(status, message)

An error status from apxchol: `:invalid_argument`, `:invalid_operator` (the
matrix violates the Laplacian/SDDM/M-matrix contract), `:no_factor_values`,
`:unsupported` (e.g. a backend this build lacks), `:out_of_memory` or
`:internal_error`.
"""
struct ApxCholError <: Exception
    status::Symbol
    message::String
end
Base.showerror(io::IO, e::ApxCholError) = print(io, "ApxCholError(:", e.status, "): ", e.message)

"""
    SolveResult

Result of one solve: the solution `x`, PCG `iterations`, the PCG's final
`relative_residual` ‖b − Ax‖/‖b‖, `converged` (`relative_residual < rtol`) and
`solve_seconds`. A solve that does not converge still returns its iterate.
"""
struct SolveResult
    x::Vector{Float64}
    iterations::Int
    relative_residual::Float64
    converged::Bool
    solve_seconds::Float64
end

"""
    BlockSolveResult

Result of a multi-right-hand-side solve: the solutions `X` (one column per
right-hand side) and, per column, `iterations`, `relative_residuals` and
`converged`.
"""
struct BlockSolveResult
    X::Matrix{Float64}
    iterations::Vector{Int}
    relative_residuals::Vector{Float64}
    converged::Vector{Bool}
end

"""
    NotConvergedError(result)

Thrown by `\\` and `ldiv!`, which cannot return a convergence flag, when the
solve did not reach its tolerance. `result` holds the final iterate.
"""
struct NotConvergedError <: Exception
    result::Union{SolveResult,BlockSolveResult}
end
function Base.showerror(io::IO, e::NotConvergedError)
    r = e.result
    if r isa SolveResult
        print(io, "NotConvergedError: relative residual ", r.relative_residual,
              " after ", r.iterations, " iterations")
    else
        print(io, "NotConvergedError: ", count(!, r.converged), " of ",
              length(r.converged), " columns did not converge")
    end
end

# ── Library loading ──────────────────────────────────────────────────────────

function read_build_info()
    info = Dict{String,String}()
    isfile(build_info_path) || return info
    for line in eachline(build_info_path)
        occursin('=', line) || continue
        key, value = split(line, '='; limit=2)
        info[key] = value
    end
    return info
end

function __init__()
    isfile(libapxchol) ||
        error("ApxChol is not built; run `import Pkg; Pkg.build(\"ApxChol\")`")
    built_against = get(read_build_info(), "libomp", "")
    if !isempty(built_against) && realpath(built_against) != realpath(LLVMOpenMP_jll.libomp_path)
        error("ApxChol was built against $(built_against), but LLVMOpenMP_jll now provides " *
              "$(LLVMOpenMP_jll.libomp_path); rebuild with `Pkg.build(\"ApxChol\")`")
    end
    # While few threads exist, libomp identifies the calling thread by its stack
    # address. Julia tasks run on their own stacks, so concurrent solves on Julia
    # threads could share one OpenMP thread identity and crash (observed with
    # Julia 1.10). Keyed thread-local lookup (mode 2, available in every libomp
    # build) does not depend on the stack; it must be chosen before libomp's
    # first OpenMP call. A caller's setting wins.
    haskey(ENV, "KMP_GTID_MODE") || (ENV["KMP_GTID_MODE"] = "2")
    # The OpenMP runtime the library links goes in first: two OpenMP runtimes
    # in one process can crash once a parallel region starts.
    Libdl.dlopen(LLVMOpenMP_jll.libomp_path, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    Libdl.dlopen(libapxchol, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    abi = ccall((:apxchol_c_abi_version, libapxchol), Int32, ())
    abi == C_ABI_VERSION ||
        error("libapxchol has C ABI version $abi, expected $C_ABI_VERSION; rebuild ApxChol")
    return nothing
end

"""
    ApxChol.version() -> String

The apxchol version and git revision the library was built from.
"""
version() = unsafe_string(ccall((:apxchol_version, libapxchol), Cstring, ()))

"""
    ApxChol.openmp_enabled() -> Bool
"""
openmp_enabled() = ccall((:apxchol_openmp_enabled, libapxchol), Int32, ()) == 1

"""
    ApxChol.max_threads() -> Int

The OpenMP team limit of the calling thread (1 in a serial build). Solvers
created with `threads=0` use it.
"""
max_threads() = Int(ccall((:apxchol_get_max_threads, libapxchol), Int32, ()))

"""
    ApxChol.backend_available(backend::Symbol) -> Bool

Whether this build supports `backend` (`:cpu`, or `:metal` on Apple silicon
builds with a usable GPU).
"""
backend_available(backend::Symbol) =
    backend in BACKENDS &&
    ccall((:apxchol_backend_available, libapxchol), Int32, (Int32,),
          enum_code(BACKENDS, backend, "backend")) == 1

const ERROR_CAPACITY = 2048

error_text(buffer::Vector{UInt8}) =
    String(buffer[1:something(findfirst(iszero, buffer), length(buffer) + 1) - 1])

function check(status::Integer, buffer::Vector{UInt8})
    status == 0 && return nothing
    throw(ApxCholError(status_symbol(status), error_text(buffer)))
end

# ── Options ──────────────────────────────────────────────────────────────────

function default_options()
    options = Ref{COptions}()
    status = ccall((:apxchol_options_default, libapxchol), Int32, (Ref{COptions}, Csize_t),
                   options, sizeof(COptions))
    status == 0 || error("apxchol_options_default failed with status $status")
    return options[]
end

int32(value::Integer, what) =
    typemin(Int32) <= value <= typemax(Int32) ? Int32(value) :
    throw(ArgumentError("$what = $value does not fit in Int32"))

function make_options(; rtol, maxiter, stagnation_window, seed, threads, sampler,
                      partitioner, storage, degree_quantile, exact_clique_max_degree,
                      keep_factor, backend)
    d = default_options()
    return COptions(
        d.struct_size,
        enum_code(BACKENDS, backend, "backend"),
        rtol === nothing ? d.tol : Float64(rtol),
        maxiter === nothing ? d.max_iter : int32(maxiter, "maxiter"),
        stagnation_window === nothing ? d.stagnation_window :
            int32(stagnation_window, "stagnation_window"),
        seed === nothing ? d.seed :
            (0 <= seed <= typemax(UInt32) ? UInt32(seed) :
             throw(ArgumentError("seed = $seed does not fit in UInt32"))),
        int32(threads, "threads"),
        sampler === nothing ? d.sampler : enum_code(SAMPLERS, sampler, "sampler"),
        partitioner === nothing ? d.partitioner :
            enum_code(PARTITIONERS, partitioner, "partitioner"),
        storage === nothing ? d.storage : enum_code(STORAGES, storage, "storage"),
        Int32(keep_factor),
        degree_quantile === nothing ? d.degree_quantile : Float64(degree_quantile),
        exact_clique_max_degree === nothing ? d.exact_clique_max_degree :
            (exact_clique_max_degree >= 0 ? UInt64(exact_clique_max_degree) :
             throw(ArgumentError("exact_clique_max_degree must be non-negative"))))
end

# ── Solver handle ────────────────────────────────────────────────────────────

"""
    ApxCholSolver(A::SparseMatrixCSC; kwargs...)

Factorize `A` once for repeated solves. `A` must be a Laplacian, SDDM or
lumpable M-matrix (symmetric, positive diagonal, non-positive off-diagonals
up to apxchol's lumping); it is validated, never reinterpreted. Laplacian
systems return the min-norm solution and need a right-hand side compatible
with every connected component.

Keywords (`nothing` selects apxchol's default):
- `rtol` (1e-8), `maxiter` (200), `stagnation_window` (50; 0 disables):
  defaults for each solve;
- `seed` (42), `sampler` (`:gks` or `:trace_cycle`), `partitioner`
  (`:block_greedy`, `:priority_greedy`, `:baumann_kyng`), `storage`
  (`:vec_pool_aos`, `:vec`, `:bstr`), `degree_quantile`,
  `exact_clique_max_degree`: factorization options;
- `threads = 0`: OpenMP threads for every call on this solver (0 inherits the
  calling thread's setting);
- `keep_factor = false`: retain the factor values for [`ApxChol.factor`](@ref);
- `backend = :cpu`.

The solver holds native memory released by its finalizer or by `close`.
Calls on one solver are serialized; distinct solvers may be used concurrently.
"""
mutable struct ApxCholSolver
    ptr::Ptr{Cvoid}
    const n::Int
    const options::COptions
    const lock::ReentrantLock
end

function ApxCholSolver(A::SparseMatrixCSC; rtol=nothing, maxiter=nothing,
                       stagnation_window=nothing, seed=nothing, threads::Integer=0,
                       sampler=nothing, partitioner=nothing, storage=nothing,
                       degree_quantile=nothing, exact_clique_max_degree=nothing,
                       keep_factor::Bool=false, backend::Symbol=:cpu)
    n = LinearAlgebra.checksquare(A)
    n >= 1 || throw(ArgumentError("A must have at least one row"))
    threads >= 0 || throw(ArgumentError("threads must be non-negative"))
    options = make_options(; rtol, maxiter, stagnation_window, seed, threads, sampler,
                           partitioner, storage, degree_quantile, exact_clique_max_degree,
                           keep_factor, backend)
    M = A isa SparseMatrixCSC{Float64,Int64} ? A : SparseMatrixCSC{Float64,Int64}(A)
    handle = Ref{Ptr{Cvoid}}(C_NULL)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = GC.@preserve M ccall(
        (:apxchol_solver_create, libapxchol), Int32,
        (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Int32, Ref{COptions},
         Ref{Ptr{Cvoid}}, Ptr{UInt8}, Csize_t),
        n, M.colptr, M.rowval, M.nzval, 1, options, handle, buffer, length(buffer))
    check(status, buffer)
    solver = ApxCholSolver(handle[], n, options, ReentrantLock())
    finalizer(destroy!, solver)
    return solver
end

# The finalizer runs only once the solver is unreachable, so no call can race it.
function destroy!(S::ApxCholSolver)
    if S.ptr != C_NULL
        ccall((:apxchol_solver_destroy, libapxchol), Cvoid, (Ptr{Cvoid},), S.ptr)
        S.ptr = C_NULL
    end
    return nothing
end

"""
    close(S::ApxCholSolver)

Release the solver's native memory now. Further use throws; closing twice is
a no-op.
"""
Base.close(S::ApxCholSolver) = lock(() -> destroy!(S), S.lock)
Base.isopen(S::ApxCholSolver) = S.ptr != C_NULL
Base.size(S::ApxCholSolver) = (S.n, S.n)
Base.size(S::ApxCholSolver, d::Integer) = d <= 2 ? S.n : 1
Base.eltype(::Type{ApxCholSolver}) = Float64

function Base.show(io::IO, S::ApxCholSolver)
    print(io, "ApxCholSolver(", S.n, "×", S.n, ", ",
          enum_symbol(BACKENDS, S.options.backend), isopen(S) ? "" : ", closed", ")")
end

function with_handle(f, S::ApxCholSolver)
    lock(S.lock) do
        S.ptr == C_NULL && throw(ArgumentError("the ApxCholSolver has been closed"))
        f(S.ptr)
    end
end

dense_vector(v::Vector{Float64}) = v
dense_vector(v::AbstractVector{<:Real}) = Vector{Float64}(v)
dense_matrix(M::Matrix{Float64}) = M
dense_matrix(M::AbstractMatrix{<:Real}) = Matrix{Float64}(M)

call_tol(rtol) = rtol === nothing ? -1.0 : Float64(rtol)
call_maxiter(maxiter) = maxiter === nothing ? Int32(-1) :
    maxiter >= 0 ? int32(maxiter, "maxiter") : throw(ArgumentError("maxiter must be non-negative"))

function check_length(v, n, what)
    length(v) == n || throw(DimensionMismatch("$what has length $(length(v)), expected $n"))
end

"""
    solve!(x, S::ApxCholSolver, b; x0=nothing, rtol=nothing, maxiter=nothing) -> SolveResult
    solve!(X, S::ApxCholSolver, B; X0=nothing, rtol=nothing, maxiter=nothing) -> BlockSolveResult

Solve `A x = b` (or every column of `A X = B`) into the preallocated
`Vector{Float64}` `x` (`Matrix{Float64}` `X`), optionally from the initial
guess `x0` (`X0`). `rtol`/`maxiter` default to the solver's options.
Non-convergence is reported in the result, not thrown.
"""
function solve!(x::Vector{Float64}, S::ApxCholSolver, b::AbstractVector{<:Real};
                x0::Union{Nothing,AbstractVector{<:Real}}=nothing, rtol=nothing,
                maxiter=nothing)
    check_length(b, S.n, "b"); check_length(x, S.n, "x")
    x0 === nothing || check_length(x0, S.n, "x0")
    rhs = dense_vector(b)
    guess = x0 === nothing ? nothing : dense_vector(x0)
    info = Ref(CSolveInfo())
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = with_handle(S) do ptr
        GC.@preserve rhs guess x ccall(
            (:apxchol_solver_solve, libapxchol), Int32,
            (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Float64, Int32,
             Ref{CSolveInfo}, Ptr{UInt8}, Csize_t),
            ptr, rhs, guess === nothing ? C_NULL : pointer(guess), x,
            call_tol(rtol), call_maxiter(maxiter), info, buffer, length(buffer))
    end
    status == 1 || check(status, buffer)
    i = info[]
    return SolveResult(x, i.iterations, i.relative_residual, i.converged == 1, i.solve_seconds)
end

function solve!(X::Matrix{Float64}, S::ApxCholSolver, B::AbstractMatrix{<:Real};
                X0::Union{Nothing,AbstractMatrix{<:Real}}=nothing, rtol=nothing,
                maxiter=nothing)
    size(B, 1) == S.n || throw(DimensionMismatch("B has $(size(B, 1)) rows, expected $(S.n)"))
    size(X) == size(B) || throw(DimensionMismatch("X has size $(size(X)), expected $(size(B))"))
    X0 === nothing || size(X0) == size(B) ||
        throw(DimensionMismatch("X0 has size $(size(X0)), expected $(size(B))"))
    k = size(B, 2)
    rhs = dense_matrix(B)
    guess = X0 === nothing ? nothing : dense_matrix(X0)
    iterations = Vector{Int64}(undef, k)
    residuals = Vector{Float64}(undef, k)
    converged = Vector{Int32}(undef, k)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = with_handle(S) do ptr
        GC.@preserve rhs guess X iterations residuals converged ccall(
            (:apxchol_solver_solve_block, libapxchol), Int32,
            (Ptr{Cvoid}, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Float64, Int32,
             Ptr{Int64}, Ptr{Float64}, Ptr{Int32}, Ptr{UInt8}, Csize_t),
            ptr, k, rhs, guess === nothing ? C_NULL : pointer(guess), X,
            call_tol(rtol), call_maxiter(maxiter), iterations, residuals, converged,
            buffer, length(buffer))
    end
    status == 1 || check(status, buffer)
    return BlockSolveResult(X, iterations, residuals, converged .== 1)
end

"""
    solve(S::ApxCholSolver, b; x0=nothing, rtol=nothing, maxiter=nothing) -> SolveResult
    solve(S::ApxCholSolver, B::AbstractMatrix; X0=nothing, ...) -> BlockSolveResult
    solve(A::SparseMatrixCSC, b; kwargs...) -> SolveResult

Solve with an existing solver, or factorize `A` for a single solve (the
keywords of [`ApxCholSolver`](@ref) apply). Non-convergence is reported in the
result, not thrown.
"""
solve(S::ApxCholSolver, b::AbstractVector{<:Real}; kwargs...) =
    solve!(Vector{Float64}(undef, S.n), S, b; kwargs...)
solve(S::ApxCholSolver, B::AbstractMatrix{<:Real}; kwargs...) =
    solve!(Matrix{Float64}(undef, size(B)), S, B; kwargs...)

function solve(A::SparseMatrixCSC, b::AbstractVecOrMat{<:Real}; x0=nothing, kwargs...)
    S = ApxCholSolver(A; kwargs...)
    try
        return b isa AbstractVector ? solve(S, b; x0) : solve(S, b; X0=x0)
    finally
        close(S)
    end
end

function converged_or_throw(result)
    ok = result isa SolveResult ? result.converged : all(result.converged)
    ok || throw(NotConvergedError(result))
    return result
end

Base.:\(S::ApxCholSolver, b::AbstractVector{<:Real}) = converged_or_throw(solve(S, b)).x
Base.:\(S::ApxCholSolver, B::AbstractMatrix{<:Real}) = converged_or_throw(solve(S, B)).X
function LinearAlgebra.ldiv!(x::Vector{Float64}, S::ApxCholSolver, b::AbstractVector{<:Real})
    converged_or_throw(solve!(x, S, b))
    return x
end
function LinearAlgebra.ldiv!(X::Matrix{Float64}, S::ApxCholSolver, B::AbstractMatrix{<:Real})
    converged_or_throw(solve!(X, S, B))
    return X
end

# ── Preconditioner ───────────────────────────────────────────────────────────

"""
    apply!(z, S::ApxCholSolver, r) -> z

One preconditioner application `z = M⁻¹ r` (forward and backward triangular
solve with the approximate factor). For a Laplacian, `z` is defined up to the
constant vector. `z` and `r` may be the same vector.
"""
function apply!(z::Vector{Float64}, S::ApxCholSolver, r::AbstractVector{<:Real})
    check_length(r, S.n, "r"); check_length(z, S.n, "z")
    v = dense_vector(r)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = with_handle(S) do ptr
        GC.@preserve v z ccall(
            (:apxchol_solver_apply, libapxchol), Int32,
            (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{UInt8}, Csize_t),
            ptr, v, z, buffer, length(buffer))
    end
    check(status, buffer)
    return z
end

"""
    ApxCholPreconditioner

The preconditioner `M ≈ A` of an [`ApxCholSolver`](@ref):
`ldiv!(y, P, x)`, `ldiv!(P, x)` and `P \\ x` apply `M⁻¹` once. Use it as
`Krylov.cg(A, b; M=P, ldiv=true)` or `IterativeSolvers.cg(A, b; Pl=P)`.
"""
struct ApxCholPreconditioner
    solver::ApxCholSolver
end

"""
    ApxChol.aspreconditioner(S::ApxCholSolver) -> ApxCholPreconditioner
"""
aspreconditioner(S::ApxCholSolver) = ApxCholPreconditioner(S)

Base.size(P::ApxCholPreconditioner, args...) = size(P.solver, args...)
Base.eltype(::Type{ApxCholPreconditioner}) = Float64
LinearAlgebra.ldiv!(y::Vector{Float64}, P::ApxCholPreconditioner, x::AbstractVector{<:Real}) =
    apply!(y, P.solver, x)
LinearAlgebra.ldiv!(y::AbstractVector{Float64}, P::ApxCholPreconditioner, x::AbstractVector{<:Real}) =
    copyto!(y, apply!(Vector{Float64}(undef, length(y)), P.solver, x))
LinearAlgebra.ldiv!(P::ApxCholPreconditioner, x::Vector{Float64}) = apply!(x, P.solver, x)
Base.:\(P::ApxCholPreconditioner, x::AbstractVector{<:Real}) =
    apply!(Vector{Float64}(undef, length(x)), P.solver, x)

# ── Introspection ────────────────────────────────────────────────────────────

"""
    ApxChol.stats(S::ApxCholSolver) -> NamedTuple

`backend`, `n`, `factor_nnz` (entries of L including its diagonal),
`lumped_offdiag` (positive off-diagonals lumped onto the diagonal, two per
symmetric pair), `rounds`, `peak_graph_bytes`, `setup_seconds`, `sddm`
(`false` for a Laplacian) and `setup_max_threads`.
"""
function stats(S::ApxCholSolver)
    out = Ref(CStats())
    status = with_handle(S) do ptr
        ccall((:apxchol_solver_stats, libapxchol), Int32, (Ptr{Cvoid}, Ref{CStats}), ptr, out)
    end
    status == 0 || throw(ApxCholError(status_symbol(status), "apxchol_solver_stats failed"))
    s = out[]
    return (backend=enum_symbol(BACKENDS, s.backend), n=Int(s.n), factor_nnz=Int(s.factor_nnz),
            lumped_offdiag=Int(s.lumped_offdiag), rounds=Int(s.rounds),
            peak_graph_bytes=Int(s.peak_graph_bytes), setup_seconds=s.setup_seconds,
            sddm=s.sddm == 1, setup_max_threads=Int(s.setup_max_threads))
end

"""
    ApxChol.options(S::ApxCholSolver) -> NamedTuple

The resolved options the solver was created with.
"""
function options(S::ApxCholSolver)
    o = S.options
    return (backend=enum_symbol(BACKENDS, o.backend), rtol=o.tol, maxiter=Int(o.max_iter),
            stagnation_window=Int(o.stagnation_window), seed=Int(o.seed), threads=Int(o.threads),
            sampler=enum_symbol(SAMPLERS, o.sampler),
            partitioner=enum_symbol(PARTITIONERS, o.partitioner),
            storage=enum_symbol(STORAGES, o.storage), keep_factor=o.keep_factor_values == 1,
            degree_quantile=o.degree_quantile,
            exact_clique_max_degree=Int(o.exact_clique_max_degree))
end

function export_factor(S::ApxCholSolver, with_values::Bool)
    nnz_l = stats(S).factor_nnz
    colptr = with_values ? Vector{Int64}(undef, S.n + 1) : nothing
    rowval = with_values ? Vector{Int64}(undef, nnz_l) : nothing
    nzval = with_values ? Vector{Float64}(undef, nnz_l) : nothing
    perm = Vector{Int64}(undef, S.n)
    ptr_or_null(v) = v === nothing ? C_NULL : pointer(v)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = with_handle(S) do ptr
        GC.@preserve colptr rowval nzval perm ccall(
            (:apxchol_solver_export_factor, libapxchol), Int32,
            (Ptr{Cvoid}, Int32, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64},
             Ptr{UInt8}, Csize_t),
            ptr, 1, ptr_or_null(colptr), ptr_or_null(rowval), ptr_or_null(nzval), perm,
            buffer, length(buffer))
    end
    check(status, buffer)
    return colptr, rowval, nzval, perm
end

"""
    ApxChol.factor(S::ApxCholSolver) -> (L, perm)

The approximate factor: lower-triangular `L::SparseMatrixCSC{Float64,Int64}`
in the elimination order (fp32 values widened) and `perm`, where `perm[v]` is
vertex `v`'s elimination position, so `A[p, p] ≈ L * L'` for
`p = invperm(perm)`. Requires `keep_factor=true`.
"""
function factor(S::ApxCholSolver)
    colptr, rowval, nzval, perm = export_factor(S, true)
    return SparseMatrixCSC(S.n, S.n, colptr, rowval, nzval), perm
end

"""
    ApxChol.permutation(S::ApxCholSolver) -> Vector{Int}

The elimination order (`perm[v]` = position of vertex `v`); always available.
"""
permutation(S::ApxCholSolver) = export_factor(S, false)[4]

end
