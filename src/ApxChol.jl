"""
    ApxChol

Julia bindings for [apxchol](https://github.com/AlgOptGroup/apxchol), a parallel
approximate-Cholesky preconditioned conjugate-gradient solver for graph
Laplacian, SDDM and lumpable M-matrix systems, through apxchol's C ABI.

Factor once with [`ApxCholSolver`](@ref), then [`solve`](@ref) any number of
right-hand sides; [`aspreconditioner`](@ref) exposes one preconditioner
application for external Krylov solvers. An [`ApxCholFactor`](@ref) carries one
factorization to any number of solvers, including solvers for a different
operator of the same size (factor reuse).
"""
module ApxChol

using LLVMOpenMP_jll
using Libdl
using LinearAlgebra
using SparseArrays

export ApxCholSolver, ApxCholFactor, SolveResult, BlockSolveResult, ApxCholError,
       NotConvergedError, solve, solve!

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
    omp_threshold::UInt64
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

nonnegative(value, what) =
    value >= 0 ? UInt64(value) : throw(ArgumentError("$what must be non-negative"))

# `base` (apxchol's defaults, or a factor's options) with every non-`nothing`
# keyword applied.
function make_options(base::COptions=default_options(); rtol=nothing, maxiter=nothing,
                      stagnation_window=nothing, seed=nothing, threads::Integer, sampler=nothing,
                      partitioner=nothing, storage=nothing, degree_quantile=nothing,
                      exact_clique_max_degree=nothing, omp_threshold=nothing,
                      keep_factor::Bool, backend::Symbol)
    d = base
    threads >= 0 || throw(ArgumentError("threads must be non-negative"))
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
            nonnegative(exact_clique_max_degree, "exact_clique_max_degree"),
        omp_threshold === nothing ? d.omp_threshold : nonnegative(omp_threshold, "omp_threshold"))
end

# A square, non-empty `SparseMatrixCSC{Float64,Int64}` view or copy of `A`.
function operator_csc(A::SparseMatrixCSC)
    n = LinearAlgebra.checksquare(A)
    n >= 1 || throw(ArgumentError("A must have at least one row"))
    return n, A isa SparseMatrixCSC{Float64,Int64} ? A : SparseMatrixCSC{Float64,Int64}(A)
end

# ── Solver and factor handles ────────────────────────────────────────────────

"""
    ApxCholSolver(A::SparseMatrixCSC; kwargs...)
    ApxCholSolver(A::SparseMatrixCSC, F::ApxCholFactor; rtol, maxiter, stagnation_window,
                  threads=0, keep_factor=false, backend=:cpu)

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
  `exact_clique_max_degree`, `omp_threshold` (2000: active vertices before the
  factorization engages OpenMP): factorization options;
- `threads = 0`: OpenMP threads for every call on this solver (0 inherits the
  calling thread's setting);
- `keep_factor = false`: retain the factor values for [`ApxChol.factor`](@ref)
  and [`ApxCholFactor(S)`](@ref ApxCholFactor);
- `backend = :cpu`.

The second form adopts a copy of the factorization `F` instead of factorizing
(factor reuse): `A` may differ from the matrix `F` was built from, but must
have its size. It takes no factorization keywords. A Laplacian factor (rank
n − 1, centred solutions) refuses an SDDM operator; an SDDM factor may
precondition a Laplacian, whose solution is then not centred. On another
operator, convergence depends on how close it is to the factor's; check
`converged`. The first form is `ApxCholFactor(A; ...)` followed by the second
on the same matrix with the same keywords.

The solver holds native memory released by its finalizer or by `close`.
Calls on one solver are serialized; distinct solvers may be used concurrently.
"""
mutable struct ApxCholSolver
    ptr::Ptr{Cvoid}
    const n::Int
    const options::COptions
    const lock::ReentrantLock
end

"""
    ApxCholFactor(A::SparseMatrixCSC; seed, sampler, partitioner, storage,
                  degree_quantile, exact_clique_max_degree, omp_threshold, threads=0)
    ApxCholFactor(S::ApxCholSolver)

One approximate factorization, without a solver, for any number of
[`ApxCholSolver(A, F)`](@ref ApxCholSolver) to adopt. The first form factorizes
`A` with the factorization keywords of `ApxCholSolver` (and `threads` for this
call). The second copies the factor of a solver created with
`keep_factor=true` (`ApxCholError(:no_factor_values)` otherwise).

A factor always keeps its values: [`ApxChol.factor`](@ref) and
[`ApxChol.stats`](@ref) work on it. Each solver adopts its own copy, so a
factor and its solvers can be closed in any order. The factor holds native
memory released by its finalizer or by `close`; calls on one factor are
serialized.
"""
mutable struct ApxCholFactor
    ptr::Ptr{Cvoid}
    const n::Int
    const options::COptions        # factorization options only (solve fields: defaults)
    const lock::ReentrantLock
end

handle_name(::ApxCholSolver) = "ApxCholSolver"
handle_name(::ApxCholFactor) = "ApxCholFactor"

function ApxCholSolver(A::SparseMatrixCSC; rtol=nothing, maxiter=nothing,
                       stagnation_window=nothing, seed=nothing, threads::Integer=0,
                       sampler=nothing, partitioner=nothing, storage=nothing,
                       degree_quantile=nothing, exact_clique_max_degree=nothing,
                       omp_threshold=nothing, keep_factor::Bool=false, backend::Symbol=:cpu)
    n, M = operator_csc(A)
    options = make_options(; rtol, maxiter, stagnation_window, seed, threads, sampler,
                           partitioner, storage, degree_quantile, exact_clique_max_degree,
                           omp_threshold, keep_factor, backend)
    handle = Ref{Ptr{Cvoid}}(C_NULL)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = GC.@preserve M ccall(
        (:apxchol_solver_create, libapxchol), Int32,
        (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Int32, Ref{COptions},
         Ref{Ptr{Cvoid}}, Ptr{UInt8}, Csize_t),
        n, M.colptr, M.rowval, M.nzval, 1, options, handle, buffer, length(buffer))
    check(status, buffer)
    return finalizer(destroy!, ApxCholSolver(handle[], n, options, ReentrantLock()))
end

function ApxCholSolver(A::SparseMatrixCSC, F::ApxCholFactor; rtol=nothing, maxiter=nothing,
                       stagnation_window=nothing, threads::Integer=0,
                       keep_factor::Bool=false, backend::Symbol=:cpu)
    n, M = operator_csc(A)
    n == F.n || throw(DimensionMismatch("A is $n×$n, the factor $(F.n)×$(F.n)"))
    # The factor's factorization options, so `options(S)` describes its factor.
    options = make_options(F.options; rtol, maxiter, stagnation_window, threads, keep_factor,
                           backend)
    handle = Ref{Ptr{Cvoid}}(C_NULL)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = with_handle(F) do fptr
        GC.@preserve M ccall(
            (:apxchol_solver_create_from_factor, libapxchol), Int32,
            (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Int32, Ptr{Cvoid}, Ref{COptions},
             Ref{Ptr{Cvoid}}, Ptr{UInt8}, Csize_t),
            n, M.colptr, M.rowval, M.nzval, 1, fptr, options, handle, buffer, length(buffer))
    end
    check(status, buffer)
    return finalizer(destroy!, ApxCholSolver(handle[], n, options, ReentrantLock()))
end

function ApxCholFactor(A::SparseMatrixCSC; seed=nothing, threads::Integer=0, sampler=nothing,
                       partitioner=nothing, storage=nothing, degree_quantile=nothing,
                       exact_clique_max_degree=nothing, omp_threshold=nothing)
    n, M = operator_csc(A)
    options = make_options(; seed, threads, sampler, partitioner, storage, degree_quantile,
                           exact_clique_max_degree, omp_threshold, keep_factor=false,
                           backend=:cpu)
    handle = Ref{Ptr{Cvoid}}(C_NULL)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = GC.@preserve M ccall(
        (:apxchol_factor_create, libapxchol), Int32,
        (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Int32, Ref{COptions},
         Ref{Ptr{Cvoid}}, Ptr{UInt8}, Csize_t),
        n, M.colptr, M.rowval, M.nzval, 1, options, handle, buffer, length(buffer))
    check(status, buffer)
    return finalizer(destroy!, ApxCholFactor(handle[], n, options, ReentrantLock()))
end

function ApxCholFactor(S::ApxCholSolver)
    handle = Ref{Ptr{Cvoid}}(C_NULL)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = with_handle(S) do ptr
        ccall((:apxchol_solver_copy_factor, libapxchol), Int32,
              (Ptr{Cvoid}, Ref{Ptr{Cvoid}}, Ptr{UInt8}, Csize_t),
              ptr, handle, buffer, length(buffer))
    end
    check(status, buffer)
    return finalizer(destroy!, ApxCholFactor(handle[], S.n, factorization_options(S.options),
                                             ReentrantLock()))
end

# A factor carries only factorization options: `o`'s, with apxchol's defaults
# for everything a solve reads, so solvers adopting it start from those.
function factorization_options(o::COptions)
    d = default_options()
    return COptions(d.struct_size, d.backend, d.tol, d.max_iter, d.stagnation_window, o.seed,
                    o.threads, o.sampler, o.partitioner, o.storage, d.keep_factor_values,
                    o.degree_quantile, o.exact_clique_max_degree, o.omp_threshold)
end

# The finalizer runs only once the handle is unreachable, so no call can race it.
function destroy!(S::ApxCholSolver)
    if S.ptr != C_NULL
        ccall((:apxchol_solver_destroy, libapxchol), Cvoid, (Ptr{Cvoid},), S.ptr)
        S.ptr = C_NULL
    end
    return nothing
end
function destroy!(F::ApxCholFactor)
    if F.ptr != C_NULL
        ccall((:apxchol_factor_destroy, libapxchol), Cvoid, (Ptr{Cvoid},), F.ptr)
        F.ptr = C_NULL
    end
    return nothing
end

const Handle = Union{ApxCholSolver,ApxCholFactor}

"""
    close(S::ApxCholSolver)
    close(F::ApxCholFactor)

Release the handle's native memory now. Further use throws; closing twice is
a no-op. Solvers adopted from a factor keep working after it is closed.
"""
Base.close(H::Handle) = lock(() -> destroy!(H), H.lock)
Base.isopen(H::Handle) = H.ptr != C_NULL
Base.size(H::Handle) = (H.n, H.n)
Base.size(H::Handle, d::Integer) = d <= 2 ? H.n : 1
Base.eltype(::Type{ApxCholSolver}) = Float64
Base.eltype(::Type{ApxCholFactor}) = Float64

function Base.show(io::IO, S::ApxCholSolver)
    print(io, "ApxCholSolver(", S.n, "×", S.n, ", ",
          enum_symbol(BACKENDS, S.options.backend), isopen(S) ? "" : ", closed", ")")
end
Base.show(io::IO, F::ApxCholFactor) =
    print(io, "ApxCholFactor(", F.n, "×", F.n, isopen(F) ? "" : ", closed", ")")

# Every C call on a handle goes through here: under its lock, and with the
# handle object rooted so its finalizer cannot run during the call.
function with_handle(f, H::Handle)
    lock(H.lock) do
        H.ptr == C_NULL && throw(ArgumentError("the $(handle_name(H)) has been closed"))
        GC.@preserve H f(H.ptr)
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
solve with the approximate factor). `z` and `r` may be the same vector.

With an SDDM factor (`stats(S).sddm`) this is a fixed linear map. With a
Laplacian factor `z` is defined only up to a constant vector: `:metal`
centres input and output on every application, so the same `r` gives the
same `z`; `:cpu` centres them only on every K-th application of the solver's
count (apxchol's center-k schedule, `APXCHOL_CENTER_K`, default 10; every
solve restarts the count), so repeated applications of a mean-zero `r` agree
up to a constant vector.
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
`Krylov.cg(A, b; M=P, ldiv=true)` or `IterativeSolvers.cg(A, b; Pl=P)`. It is
a fixed linear map for an SDDM factor; for a Laplacian factor only up to a
constant vector (see [`ApxChol.apply!`](@ref)).
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

c_stats(::ApxCholSolver, ptr, out) =
    ccall((:apxchol_solver_stats, libapxchol), Int32, (Ptr{Cvoid}, Ref{CStats}), ptr, out)
c_stats(::ApxCholFactor, ptr, out) =
    ccall((:apxchol_factor_stats, libapxchol), Int32, (Ptr{Cvoid}, Ref{CStats}), ptr, out)

"""
    ApxChol.stats(S::ApxCholSolver) -> NamedTuple
    ApxChol.stats(F::ApxCholFactor) -> NamedTuple

`backend`, `n`, `factor_nnz` (entries of L including its diagonal),
`lumped_offdiag` (positive off-diagonals lumped onto the diagonal, two per
symmetric pair, in the matrix the factor was built from), `rounds`,
`peak_graph_bytes`, `setup_seconds` (wall time of the call that created the
handle; a copied factor reports its solver's), `sddm` (`false` for a
Laplacian factor) and `setup_max_threads`. A solver that adopted a factor
reports the factor's fields with its own setup time; a factor reports
`backend = :cpu` (factorizations run on the host).
"""
function stats(H::Handle)
    out = Ref(CStats())
    status = with_handle(H) do ptr
        c_stats(H, ptr, out)
    end
    status == 0 || throw(ApxCholError(status_symbol(status), "apxchol stats failed"))
    s = out[]
    return (backend=enum_symbol(BACKENDS, s.backend), n=Int(s.n), factor_nnz=Int(s.factor_nnz),
            lumped_offdiag=Int(s.lumped_offdiag), rounds=Int(s.rounds),
            peak_graph_bytes=Int(s.peak_graph_bytes), setup_seconds=s.setup_seconds,
            sddm=s.sddm == 1, setup_max_threads=Int(s.setup_max_threads))
end

"""
    ApxChol.options(S::ApxCholSolver) -> NamedTuple

The resolved options the solver was created with (for a solver that adopted
a factor, the factor's factorization options).
"""
function options(S::ApxCholSolver)
    o = S.options
    return (backend=enum_symbol(BACKENDS, o.backend), rtol=o.tol, maxiter=Int(o.max_iter),
            stagnation_window=Int(o.stagnation_window), seed=Int(o.seed), threads=Int(o.threads),
            sampler=enum_symbol(SAMPLERS, o.sampler),
            partitioner=enum_symbol(PARTITIONERS, o.partitioner),
            storage=enum_symbol(STORAGES, o.storage), keep_factor=o.keep_factor_values == 1,
            degree_quantile=o.degree_quantile,
            exact_clique_max_degree=Int(o.exact_clique_max_degree),
            omp_threshold=Int(o.omp_threshold))
end

c_export(::ApxCholSolver, ptr, colptr, rowval, nzval, perm, buffer) =
    ccall((:apxchol_solver_export_factor, libapxchol), Int32,
          (Ptr{Cvoid}, Int32, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64}, Ptr{UInt8},
           Csize_t), ptr, 1, colptr, rowval, nzval, perm, buffer, length(buffer))
c_export(::ApxCholFactor, ptr, colptr, rowval, nzval, perm, buffer) =
    ccall((:apxchol_factor_export, libapxchol), Int32,
          (Ptr{Cvoid}, Int32, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Ptr{Int64}, Ptr{UInt8},
           Csize_t), ptr, 1, colptr, rowval, nzval, perm, buffer, length(buffer))

function export_factor(H::Handle, with_values::Bool)
    nnz_l = stats(H).factor_nnz
    colptr = with_values ? Vector{Int64}(undef, H.n + 1) : nothing
    rowval = with_values ? Vector{Int64}(undef, nnz_l) : nothing
    nzval = with_values ? Vector{Float64}(undef, nnz_l) : nothing
    perm = Vector{Int64}(undef, H.n)
    ptr_or_null(v) = v === nothing ? C_NULL : pointer(v)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = with_handle(H) do ptr
        GC.@preserve colptr rowval nzval c_export(
            H, ptr, ptr_or_null(colptr), ptr_or_null(rowval), ptr_or_null(nzval), perm, buffer)
    end
    check(status, buffer)
    return colptr, rowval, nzval, perm
end

"""
    ApxChol.factor(S::ApxCholSolver) -> (L, perm)
    ApxChol.factor(F::ApxCholFactor) -> (L, perm)

The approximate factor: lower-triangular `L::SparseMatrixCSC{Float64,Int64}`
in the elimination order (fp32 values widened) and `perm`, where `perm[v]` is
vertex `v`'s elimination position, so `A[p, p] ≈ L * L'` for
`p = invperm(perm)`. A solver requires `keep_factor=true`; a factor always has
its values.
"""
function factor(H::Handle)
    colptr, rowval, nzval, perm = export_factor(H, true)
    return SparseMatrixCSC(H.n, H.n, colptr, rowval, nzval), perm
end

"""
    ApxChol.permutation(S::ApxCholSolver) -> Vector{Int}
    ApxChol.permutation(F::ApxCholFactor) -> Vector{Int}

The elimination order (`perm[v]` = position of vertex `v`); always available.
"""
permutation(H::Handle) = export_factor(H, false)[4]

end
