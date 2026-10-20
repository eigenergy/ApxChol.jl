# Solver and factor handles: inputs, options, construction, lifetime and the
# introspection calls.

# ── Inputs ───────────────────────────────────────────────────────────────────

const RealCSC = SparseMatrixCSC{<:Real,<:Union{Int32,Int64}}
const Operator = Union{RealCSC,Symmetric{<:Real,<:RealCSC},Hermitian{<:Real,<:RealCSC}}

# `A` as the SparseMatrixCSC{Float64,Int64} apxchol reads: `A` itself when it
# already is one, otherwise a converted copy (a symmetric wrapper stores both
# triangles). apxchol validates every index it reads, but it cannot see array
# lengths, so those are checked here before any pointer reaches native code.
operator_csc(A::Union{Symmetric,Hermitian}) = operator_csc(sparse(A))
function operator_csc(A::RealCSC)
    n = LinearAlgebra.checksquare(A)
    1 <= n <= typemax(Int32) || throw(ArgumentError("A must have between 1 and 2^31-1 rows"))
    length(A.colptr) == n + 1 || throw(ArgumentError("A.colptr must have n + 1 entries"))
    0 <= Int64(A.colptr[end]) - 1 <= min(length(A.rowval), length(A.nzval)) ||
        throw(ArgumentError("A.colptr exceeds A's row and value storage"))
    return A isa SparseMatrixCSC{Float64,Int64} ? A : SparseMatrixCSC{Float64,Int64}(A)
end

# ── Options ──────────────────────────────────────────────────────────────────

# apxchol_options_default, read once by __init__.
const DEFAULTS = Ref{COptions}()

pick(value, default) = value === nothing ? default : value

# `base` with every keyword that is not `nothing` applied (factor_storage is
# always given: `nothing` is its AUTO value). Values apxchol validates
# (positivity, ranges) are left to it; type conversions throw here.
function make_options(base::COptions; backend=nothing, rtol=nothing, maxiter=nothing,
                      stagnation_window=nothing, seed=nothing, threads=nothing, sampler=nothing,
                      partitioner=nothing, keep_factor=nothing, degree_quantile=nothing,
                      exact_clique_max_degree=nothing, omp_threshold=nothing,
                      factor_storage=storage_type(base.factor_storage))
    b = base
    return COptions(
        b.struct_size,
        backend === nothing ? b.backend : code(BACKENDS, backend, "backend"),
        Float64(pick(rtol, b.tol)),
        Int32(pick(maxiter, b.max_iter)),
        Int32(pick(stagnation_window, b.stagnation_window)),
        UInt32(pick(seed, b.seed)),
        Int32(pick(threads, b.threads)),
        sampler === nothing ? b.sampler : code(SAMPLERS, sampler, "sampler"),
        partitioner === nothing ? b.partitioner : code(PARTITIONERS, partitioner, "partitioner"),
        keep_factor === nothing ? b.keep_factor_values : Int32(keep_factor::Bool),
        Float64(pick(degree_quantile, b.degree_quantile)),
        UInt64(pick(exact_clique_max_degree, b.exact_clique_max_degree)),
        UInt64(pick(omp_threshold, b.omp_threshold)),
        storage_code(factor_storage))
end

# A factor's options: its factorization fields, with apxchol's defaults for
# everything a solve reads, so solvers adopting it start from those.
factorization_options(o::COptions) =
    make_options(DEFAULTS[]; seed=o.seed, threads=o.threads, degree_quantile=o.degree_quantile,
                 exact_clique_max_degree=o.exact_clique_max_degree, omp_threshold=o.omp_threshold,
                 sampler=symbol(SAMPLERS, o.sampler), partitioner=symbol(PARTITIONERS, o.partitioner))

# ── Handles ──────────────────────────────────────────────────────────────────

"""
    ApxCholSolver(A; backend=:auto, rtol, maxiter, stagnation_window, seed, sampler,
                  partitioner, degree_quantile, exact_clique_max_degree, omp_threshold,
                  factor_storage=nothing, threads=0, keep_factor=false)
    ApxCholSolver(A, F::ApxCholFactor; backend=:auto, rtol, maxiter, stagnation_window,
                  factor_storage=nothing, threads=0, keep_factor=false)

Factorize `A` once for any number of solves. `A` is a `SparseMatrixCSC` with
real values and `Int32` or `Int64` indices, or a `Symmetric`/`Hermitian`
wrapper of one. It must be a graph Laplacian, an SDDM matrix or a lumpable
M-matrix (symmetric, positive diagonal, nonpositive off-diagonals up to
apxchol's lumping); apxchol validates it and never reinterprets it. A
Laplacian system returns the min-norm solution and needs a right-hand side
compatible with every connected component.

Keywords left unset take apxchol's defaults (see [`ApxChol.options`](@ref)):
- `backend`: `:auto` chooses the route as apxchol's C++ solver does (CPU for at
  most 32,768 rows and when adopting a factor, otherwise Metal when available
  and the operator is within its range), `:cpu` or `:metal`;
  [`ApxChol.stats`](@ref) reports the route chosen.
- `rtol`, `maxiter`, `stagnation_window` (0 disables): every solve's defaults.
- `seed`, `sampler` (`:gks`, `:trace_cycle`), `partitioner` (`:block_greedy`,
  `:priority_greedy`, `:baumann_kyng`), `degree_quantile`,
  `exact_clique_max_degree`, `omp_threshold`: the factorization.
- `factor_storage`: the triangular solves' storage, `Float16`, `Float32` or
  `nothing` (`ENV["APXCHOL_FACTOR_STORAGE"]`, then the backend's default).
- `threads`: the OpenMP team of every call on this solver (0 inherits the
  calling thread's limit).
- `keep_factor`: retain the factor's values for [`ApxChol.factor`](@ref) and
  [`ApxCholFactor(S)`](@ref ApxCholFactor).

The second form adopts a copy of the factorization `F` instead of factorizing
(factor reuse). `A` may differ from the matrix `F` was built from but must
have its size. A Laplacian factor refuses an SDDM operator; an SDDM factor may
precondition a Laplacian, whose solution is then not centred. Convergence on
another operator depends on its distance from the factor's; check `converged`.

The solver owns native memory, released by `close` or its finalizer. Calls on
one solver are serialized; distinct solvers run concurrently.
"""
mutable struct ApxCholSolver
    ptr::Ptr{Cvoid}
    const n::Int
    const options::COptions           # as requested (backend may be :auto)
    const backend::Symbol             # the route chosen
    const lock::ReentrantLock         # held for every native call
    const info::Base.RefValue{CSolveInfo}  # written only under `lock`
    const error_buffer::Vector{UInt8}      # written only under `lock`
end

"""
    ApxCholFactor(A; seed, sampler, partitioner, degree_quantile,
                  exact_clique_max_degree, omp_threshold, threads=0)
    ApxCholFactor(S::ApxCholSolver)

One approximate factorization, without a solver, for any number of
[`ApxCholSolver(A, F)`](@ref ApxCholSolver) to adopt. The first form factorizes
`A` with the factorization keywords of `ApxCholSolver`; the second copies the
factor of a solver created with `keep_factor=true`.

A factor is immutable and always keeps its values. Solvers adopt their own
copy, so a factor and its solvers can be closed in any order. Reads of one
factor (adoptions, [`ApxChol.stats`](@ref), [`ApxChol.factor`](@ref)) run
concurrently; `close` waits for them.
"""
mutable struct ApxCholFactor
    ptr::Ptr{Cvoid}
    const n::Int
    const options::COptions           # factorization fields; solve fields are defaults
    const state::Threads.Condition    # guards ptr and readers
    readers::Int
end

const Handle = Union{ApxCholSolver,ApxCholFactor}

closed_error(H::Handle) = InvalidStateException("the $(nameof(typeof(H))) has been closed", :closed)

function solver_handle(ptr::Ptr{Cvoid}, n::Int, options::COptions, buffer::Vector{UInt8})
    route = symbol(BACKENDS, solver_stats(ptr).backend)
    return finalizer(destroy!, ApxCholSolver(ptr, n, options, route, ReentrantLock(),
                                             Ref(CSolveInfo()), buffer))
end
factor_handle(ptr::Ptr{Cvoid}, n::Int, options::COptions) =
    finalizer(destroy!, ApxCholFactor(ptr, n, options, Threads.Condition(), 0))

function ApxCholSolver(A::Operator; backend::Symbol=:auto, rtol=nothing, maxiter=nothing,
                       stagnation_window=nothing, seed=nothing, sampler=nothing,
                       partitioner=nothing, degree_quantile=nothing,
                       exact_clique_max_degree=nothing, omp_threshold=nothing,
                       factor_storage=nothing, threads::Integer=0, keep_factor::Bool=false)
    M = operator_csc(A)
    o = make_options(DEFAULTS[]; backend, rtol, maxiter, stagnation_window, seed, sampler,
                     partitioner, degree_quantile, exact_clique_max_degree, omp_threshold,
                     factor_storage, threads, keep_factor)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = GC.@preserve M @native libapxchol.apxchol_solver_create(
        size(M, 1)::Int64, M.colptr::Ptr{Int64}, M.rowval::Ptr{Int64}, M.nzval::Ptr{Float64},
        1::Int32, Ref(o)::Ref{COptions}, out::Ref{Ptr{Cvoid}}, buffer::Ptr{UInt8},
        length(buffer)::Csize_t)::Int32
    check(status, buffer)
    return solver_handle(out[], size(M, 1), o, buffer)
end

function ApxCholSolver(A::Operator, F::ApxCholFactor; backend::Symbol=:auto, rtol=nothing,
                       maxiter=nothing, stagnation_window=nothing, factor_storage=nothing,
                       threads::Integer=0, keep_factor::Bool=false)
    M = operator_csc(A)
    size(M, 1) == F.n || throw(DimensionMismatch("A is $(size(M, 1))×$(size(M, 1)), the factor $(F.n)×$(F.n)"))
    o = make_options(F.options; backend, rtol, maxiter, stagnation_window, factor_storage,
                     threads, keep_factor)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = reading(F) do fptr
        GC.@preserve M @native libapxchol.apxchol_solver_create_from_factor(
            size(M, 1)::Int64, M.colptr::Ptr{Int64}, M.rowval::Ptr{Int64},
            M.nzval::Ptr{Float64}, 1::Int32, fptr::Ptr{Cvoid}, Ref(o)::Ref{COptions},
            out::Ref{Ptr{Cvoid}}, buffer::Ptr{UInt8}, length(buffer)::Csize_t)::Int32
    end
    check(status, buffer)
    return solver_handle(out[], size(M, 1), o, buffer)
end

function ApxCholFactor(A::Operator; seed=nothing, sampler=nothing, partitioner=nothing,
                       degree_quantile=nothing, exact_clique_max_degree=nothing,
                       omp_threshold=nothing, threads::Integer=0)
    M = operator_csc(A)
    o = make_options(DEFAULTS[]; seed, sampler, partitioner, degree_quantile,
                     exact_clique_max_degree, omp_threshold, threads)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    buffer = zeros(UInt8, ERROR_CAPACITY)
    status = GC.@preserve M @native libapxchol.apxchol_factor_create(
        size(M, 1)::Int64, M.colptr::Ptr{Int64}, M.rowval::Ptr{Int64}, M.nzval::Ptr{Float64},
        1::Int32, Ref(o)::Ref{COptions}, out::Ref{Ptr{Cvoid}}, buffer::Ptr{UInt8},
        length(buffer)::Csize_t)::Int32
    check(status, buffer)
    return factor_handle(out[], size(M, 1), o)
end

function ApxCholFactor(S::ApxCholSolver)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    ptr = acquire(S)
    try
        status = GC.@preserve S @native libapxchol.apxchol_solver_copy_factor(
            ptr::Ptr{Cvoid}, out::Ref{Ptr{Cvoid}}, S.error_buffer::Ptr{UInt8},
            ERROR_CAPACITY::Csize_t)::Int32
        check(status, S.error_buffer)
    finally
        unlock(S.lock)
    end
    return factor_handle(out[], S.n, factorization_options(S.options))
end

# ── Lifetime ─────────────────────────────────────────────────────────────────

# Called by close (which excludes every in-flight call) and by the finalizer
# (which runs only once the handle is unreachable, so no call is in flight).
function destroy!(S::ApxCholSolver)
    ptr, S.ptr = S.ptr, C_NULL
    ptr == C_NULL || @ccall libapxchol.apxchol_solver_destroy(ptr::Ptr{Cvoid})::Cvoid
    return nothing
end
function destroy!(F::ApxCholFactor)
    ptr, F.ptr = F.ptr, C_NULL
    ptr == C_NULL || @ccall libapxchol.apxchol_factor_destroy(ptr::Ptr{Cvoid})::Cvoid
    return nothing
end

"""
    close(S::ApxCholSolver)
    close(F::ApxCholFactor)

Release the handle's native memory now, after any call in flight on it.
Later calls throw `InvalidStateException`; closing twice is a no-op. Solvers
that adopted a factor keep working after it is closed.
"""
Base.close(S::ApxCholSolver) = lock(() -> destroy!(S), S.lock)
function Base.close(F::ApxCholFactor)
    lock(F.state) do
        while F.readers > 0
            wait(F.state)
        end
        destroy!(F)
    end
end
Base.isopen(H::Handle) = H.ptr != C_NULL

# Locks the solver for one native call and returns its pointer; the caller
# unlocks in a `finally` and keeps the solver rooted across the call.
@inline function acquire(S::ApxCholSolver)
    lock(S.lock)
    S.ptr == C_NULL && (unlock(S.lock); throw(closed_error(S)))
    return S.ptr
end

# Runs f(ptr) as one of the factor's concurrent readers.
function reading(f, F::ApxCholFactor)
    ptr = lock(F.state) do
        F.ptr == C_NULL && throw(closed_error(F))
        F.readers += 1
        F.ptr
    end
    try
        return GC.@preserve F f(ptr)
    finally
        lock(F.state) do
            F.readers -= 1
            F.readers == 0 && notify(F.state)
        end
    end
end

Base.size(H::Handle) = (H.n, H.n)
Base.size(H::Handle, d::Integer) = d < 1 ? throw(ArgumentError("dimension must be ≥ 1")) :
                                   d <= 2 ? H.n : 1
Base.eltype(::Type{<:Handle}) = Float64

Base.show(io::IO, S::ApxCholSolver) =
    print(io, "ApxCholSolver(", S.n, "×", S.n, ", ", S.backend, isopen(S) ? "" : ", closed", ")")
Base.show(io::IO, F::ApxCholFactor) =
    print(io, "ApxCholFactor(", F.n, "×", F.n, isopen(F) ? "" : ", closed", ")")

# ── Introspection ────────────────────────────────────────────────────────────

function solver_stats(ptr::Ptr{Cvoid})
    out = Ref(CStats())
    status = @native libapxchol.apxchol_solver_stats(ptr::Ptr{Cvoid}, out::Ref{CStats})::Int32
    status == 0 || throw(ApxCholError(status_symbol(status), "apxchol_solver_stats failed"))
    return out[]
end
function factor_stats(ptr::Ptr{Cvoid})
    out = Ref(CStats())
    status = @native libapxchol.apxchol_factor_stats(ptr::Ptr{Cvoid}, out::Ref{CStats})::Int32
    status == 0 || throw(ApxCholError(status_symbol(status), "apxchol_factor_stats failed"))
    return out[]
end

stats_tuple(s::CStats) =
    (backend=symbol(BACKENDS, s.backend), n=Int(s.n), factor_nnz=Int(s.factor_nnz),
     lumped_offdiag=Int(s.lumped_offdiag), rounds=Int(s.rounds),
     peak_graph_bytes=Int(s.peak_graph_bytes), setup_seconds=s.setup_seconds,
     sddm=s.sddm == 1, setup_max_threads=Int(s.setup_max_threads),
     factor_storage=storage_type(s.factor_storage))

"""
    ApxChol.stats(S::ApxCholSolver) -> NamedTuple
    ApxChol.stats(F::ApxCholFactor) -> NamedTuple

- `backend`: the route chosen, `:cpu` or `:metal` (a factor reports `:cpu`);
- `n`, `factor_nnz` (entries of L, diagonal included), `rounds`,
  `peak_graph_bytes`;
- `lumped_offdiag`: positive off-diagonals lumped onto the diagonal (two per
  symmetric pair) in the matrix the factor was built from;
- `setup_seconds`, `setup_max_threads`: of the call that created the handle;
- `sddm`: `false` for a Laplacian factor (min-norm solutions);
- `factor_storage`: the triangular solves' storage, `Float16` or `Float32`
  (a factor reports its native `Float32`).
"""
function stats(S::ApxCholSolver)
    ptr = acquire(S)
    try
        return stats_tuple(GC.@preserve S solver_stats(ptr))
    finally
        unlock(S.lock)
    end
end
stats(F::ApxCholFactor) = stats_tuple(reading(factor_stats, F))

"""
    ApxChol.options(S::ApxCholSolver) -> NamedTuple
    ApxChol.options(F::ApxCholFactor) -> NamedTuple

The options a solver was created with (`backend` as requested; a solver that
adopted a factor reports the factor's factorization options), or a factor's
factorization options.
"""
function options(S::ApxCholSolver)
    o = S.options
    return (backend=symbol(BACKENDS, o.backend), rtol=o.tol, maxiter=Int(o.max_iter),
            stagnation_window=Int(o.stagnation_window), factor_storage=storage_type(o.factor_storage),
            threads=Int(o.threads), keep_factor=o.keep_factor_values == 1,
            factorization_tuple(o)...)
end
options(F::ApxCholFactor) = (threads=Int(F.options.threads), factorization_tuple(F.options)...)
factorization_tuple(o::COptions) =
    (seed=Int(o.seed), sampler=symbol(SAMPLERS, o.sampler),
     partitioner=symbol(PARTITIONERS, o.partitioner), degree_quantile=o.degree_quantile,
     exact_clique_max_degree=Int(o.exact_clique_max_degree), omp_threshold=Int(o.omp_threshold))

"""
    ApxChol.ExportedFactor

The approximate factor of [`ApxChol.factor`](@ref): `L`, lower triangular with
the diagonal first in each column (`SparseMatrixCSC{Float32,Int64}`, apxchol's
native values), and the elimination order `p`, so that `A[p, p] ≈ L * L'`.
"""
struct ExportedFactor
    L::SparseMatrixCSC{Float32,Int64}
    p::Vector{Int}
end

Base.:(==)(a::ExportedFactor, b::ExportedFactor) = a.p == b.p && a.L == b.L
Base.hash(f::ExportedFactor, h::UInt) = hash(f.L, hash(f.p, hash(ExportedFactor, h)))

# perm[v] is v's elimination position (C ABI); p = invperm(perm).
function export_arrays(H::Handle, ptr::Ptr{Cvoid}, buffer::Vector{UInt8}, values::Bool)
    nnz = values ? Int((H isa ApxCholSolver ? solver_stats(ptr) : factor_stats(ptr)).factor_nnz) : 0
    colptr = values ? Vector{Int64}(undef, H.n + 1) : Int64[]
    rowval = Vector{Int64}(undef, nnz)
    nzval = Vector{Float32}(undef, nnz)
    perm = Vector{Int64}(undef, H.n)
    null(v) = values ? pointer(v) : Ptr{eltype(v)}(C_NULL)
    status = GC.@preserve colptr rowval nzval perm if H isa ApxCholSolver
        @native libapxchol.apxchol_solver_export_factor(
            ptr::Ptr{Cvoid}, 1::Int32, null(colptr)::Ptr{Int64}, null(rowval)::Ptr{Int64},
            null(nzval)::Ptr{Float32}, perm::Ptr{Int64}, buffer::Ptr{UInt8},
            length(buffer)::Csize_t)::Int32
    else
        @native libapxchol.apxchol_factor_export(
            ptr::Ptr{Cvoid}, 1::Int32, null(colptr)::Ptr{Int64}, null(rowval)::Ptr{Int64},
            null(nzval)::Ptr{Float32}, perm::Ptr{Int64}, buffer::Ptr{UInt8},
            length(buffer)::Csize_t)::Int32
    end
    check(status, buffer)
    p = invperm(perm)
    return values ? ExportedFactor(SparseMatrixCSC(H.n, H.n, colptr, rowval, nzval), p) : p
end

function export_from(H::ApxCholSolver, values::Bool)
    ptr = acquire(H)
    try
        return GC.@preserve H export_arrays(H, ptr, H.error_buffer, values)
    finally
        unlock(H.lock)
    end
end
export_from(H::ApxCholFactor, values::Bool) =
    reading(ptr -> export_arrays(H, ptr, zeros(UInt8, ERROR_CAPACITY), values), H)

"""
    ApxChol.factor(S::ApxCholSolver) -> ExportedFactor
    ApxChol.factor(F::ApxCholFactor) -> ExportedFactor

The approximate factor `L` and elimination order `p` with `A[p, p] ≈ L * L'`
(see [`ApxChol.ExportedFactor`](@ref)). A solver needs `keep_factor=true`
(`ApxCholError(:no_factor_values)` otherwise); a factor always has its values.
"""
factor(H::Handle) = export_from(H, true)

"""
    ApxChol.ordering(S::ApxCholSolver) -> Vector{Int}
    ApxChol.ordering(F::ApxCholFactor) -> Vector{Int}

The elimination order `p` of [`ApxChol.factor`](@ref), always available.
"""
ordering(H::Handle) = export_from(H, false)
