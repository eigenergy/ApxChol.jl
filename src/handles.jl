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

"""
The options of one handle, as deps/apxchol_jl.cpp takes them (in this order).
"""
struct Options
    backend::Int32
    tol::Float64
    max_iter::Int32
    seed::UInt32
    threads::Int32
    sampler::Int32
    partitioner::Int32
    keep_factor_values::Int32
    degree_quantile::Float64
    exact_clique_max_degree::UInt64
    omp_threshold::UInt64
    factor_storage::Int32
end
fields(o::Options) = ntuple(i -> getfield(o, i), Val(fieldcount(Options)))

# apxchol::solve_options' defaults, read once by __init__.
const DEFAULTS = Ref{Options}()

pick(value, default) = value === nothing ? default : value

# `base` with every keyword that is not `nothing` applied (factor_storage is
# always given: `nothing` is its automatic value). Values apxchol validates
# (positivity, ranges) are left to it; type conversions throw here.
function make_options(base::Options; backend=nothing, rtol=nothing, maxiter=nothing,
                      seed=nothing, threads=nothing, sampler=nothing, partitioner=nothing,
                      keep_factor=nothing, degree_quantile=nothing,
                      exact_clique_max_degree=nothing, omp_threshold=nothing,
                      factor_storage=storage_type(base.factor_storage))
    b = base
    return Options(
        backend === nothing ? b.backend : code(BACKENDS, backend, "backend"),
        Float64(pick(rtol, b.tol)),
        Int32(pick(maxiter, b.max_iter)),
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
factorization_options(o::Options) =
    make_options(DEFAULTS[]; seed=o.seed, threads=o.threads, degree_quantile=o.degree_quantile,
                 exact_clique_max_degree=o.exact_clique_max_degree, omp_threshold=o.omp_threshold,
                 sampler=symbol(SAMPLERS, o.sampler), partitioner=symbol(PARTITIONERS, o.partitioner))

# ── Handles ──────────────────────────────────────────────────────────────────

"""
    ApxCholSolver(A; backend=:auto, rtol, maxiter, seed, sampler, partitioner,
                  degree_quantile, exact_clique_max_degree, omp_threshold,
                  factor_storage=nothing, threads=0, keep_factor=false)
    ApxCholSolver(A, F::ApxCholFactor; backend=:auto, rtol, maxiter,
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
  most 32,768 rows, when adopting a factor, and for operators or options the
  Metal setup cannot take; otherwise Metal when available), `:cpu` or `:metal`
  (`ApxCholError(:unsupported)` for what Metal cannot take);
  [`ApxChol.stats`](@ref) reports the route chosen. A Metal solver builds its
  factor on the GPU, so it differs from the CPU route's for the same seed.
- `rtol`, `maxiter`: every solve's defaults. A solve stops at the tolerance,
  the iteration limit or a numerical breakdown.
- `seed`, `sampler` (`:gks`, `:trace_cycle`), `partitioner` (`:block_greedy`,
  `:priority_greedy`, `:baumann_kyng`), `degree_quantile`,
  `exact_clique_max_degree`, `omp_threshold`: the factorization
  (`degree_quantile = -1` chooses the route's own cap).
- `factor_storage`: the triangular solves' storage, `Float16`, `Float32` or
  `nothing` (the backend's default, `Float32` on the CPU and Metal).
- `threads`: a ceiling on apxchol's OpenMP workers in every call on this
  solver (0 inherits the calling thread's limit).
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
    ptr::SolverPtr
    const n::Int
    const options::Options            # as requested (backend may be :auto)
    const backend::Symbol             # the route chosen
    const lock::ReentrantLock         # held for every native call
    # One solve's iterations, residual and convergence, written only under `lock`.
    const iterations::Vector{Int64}
    const residuals::Vector{Float64}
    const converged::Vector{Int32}
end

"""
    ApxCholFactor(A; seed, sampler, partitioner, degree_quantile,
                  exact_clique_max_degree, omp_threshold, threads=0)
    ApxCholFactor(S::ApxCholSolver)

One approximate factorization, without a solver, for any number of
[`ApxCholSolver(A, F)`](@ref ApxCholSolver) to adopt. The first form factorizes
`A` on the CPU with the factorization keywords of `ApxCholSolver`; the second
copies the factor of a solver created with `keep_factor=true`.

A factor is immutable and always keeps its values. Solvers adopt their own
copy, so a factor and its solvers can be closed in any order. Reads of one
factor (adoptions, [`ApxChol.stats`](@ref), [`ApxChol.factor`](@ref)) run
concurrently; `close` waits for them.
"""
mutable struct ApxCholFactor
    ptr::FactorPtr
    const n::Int
    const options::Options            # factorization fields; solve fields are defaults
    const state::Threads.Condition    # guards ptr and readers
    readers::Int
end

const Handle = Union{ApxCholSolver,ApxCholFactor}

closed_error(H::Handle) = InvalidStateException("the $(nameof(typeof(H))) has been closed", :closed)

function solver_handle(ptr::SolverPtr, n::Int, options::Options)
    route = symbol(BACKENDS, native_stats(ptr).backend)
    return finalizer(destroy!, ApxCholSolver(ptr, n, options, route, ReentrantLock(),
                                             zeros(Int64, 1), zeros(1), zeros(Int32, 1)))
end
factor_handle(ptr::FactorPtr, n::Int, options::Options) =
    finalizer(destroy!, ApxCholFactor(ptr, n, options, Threads.Condition(), 0))

function ApxCholSolver(A::Operator; backend::Symbol=:auto, rtol=nothing, maxiter=nothing,
                       seed=nothing, sampler=nothing, partitioner=nothing,
                       degree_quantile=nothing, exact_clique_max_degree=nothing,
                       omp_threshold=nothing, factor_storage=nothing, threads::Integer=0,
                       keep_factor::Bool=false)
    M = operator_csc(A)
    o = make_options(DEFAULTS[]; backend, rtol, maxiter, seed, sampler, partitioner,
                     degree_quantile, exact_clique_max_degree, omp_threshold, factor_storage,
                     threads, keep_factor)
    ptr = GC.@preserve M @native Native.create_solver(
        size(M, 1), pointer(M.colptr), pointer(M.rowval), pointer(M.nzval), NO_FACTOR, fields(o)...)
    return solver_handle(ptr, size(M, 1), o)
end

function ApxCholSolver(A::Operator, F::ApxCholFactor; backend::Symbol=:auto, rtol=nothing,
                       maxiter=nothing, factor_storage=nothing, threads::Integer=0,
                       keep_factor::Bool=false)
    M = operator_csc(A)
    size(M, 1) == F.n || throw(DimensionMismatch("A is $(size(M, 1))×$(size(M, 1)), the factor $(F.n)×$(F.n)"))
    o = make_options(F.options; backend, rtol, maxiter, factor_storage, threads, keep_factor)
    ptr = reading(F) do fptr
        GC.@preserve M @native Native.create_solver(
            size(M, 1), pointer(M.colptr), pointer(M.rowval), pointer(M.nzval), fptr, fields(o)...)
    end
    return solver_handle(ptr, size(M, 1), o)
end

function ApxCholFactor(A::Operator; seed=nothing, sampler=nothing, partitioner=nothing,
                       degree_quantile=nothing, exact_clique_max_degree=nothing,
                       omp_threshold=nothing, threads::Integer=0)
    M = operator_csc(A)
    o = make_options(DEFAULTS[]; seed, sampler, partitioner, degree_quantile,
                     exact_clique_max_degree, omp_threshold, threads)
    ptr = GC.@preserve M @native Native.create_factor(
        size(M, 1), pointer(M.colptr), pointer(M.rowval), pointer(M.nzval), o.seed, o.threads,
        o.sampler, o.partitioner, o.degree_quantile, o.exact_clique_max_degree, o.omp_threshold)
    return factor_handle(ptr, size(M, 1), o)
end

function ApxCholFactor(S::ApxCholSolver)
    ptr = acquire(S)
    fptr = try
        GC.@preserve S @native Native.copy_factor(ptr)
    finally
        unlock(S.lock)
    end
    return factor_handle(fptr, S.n, factorization_options(S.options))
end

# ── Lifetime ─────────────────────────────────────────────────────────────────

# Called by close (which excludes every in-flight call) and by the finalizer
# (which runs only once the handle is unreachable, so no call is in flight).
function destroy!(S::ApxCholSolver)
    ptr, S.ptr = S.ptr, SolverPtr(C_NULL)
    isnull(ptr) || Native.destroy_solver(ptr)
    return nothing
end
function destroy!(F::ApxCholFactor)
    ptr, F.ptr = F.ptr, FactorPtr(C_NULL)
    isnull(ptr) || Native.destroy_factor(ptr)
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
Base.isopen(H::Handle) = !isnull(H.ptr)

# Locks the solver for one native call and returns its pointer; the caller
# unlocks in a `finally` and keeps the solver rooted across the call.
@inline function acquire(S::ApxCholSolver)
    lock(S.lock)
    isnull(S.ptr) && (unlock(S.lock); throw(closed_error(S)))
    return S.ptr
end

# Runs f(ptr) as one of the factor's concurrent readers.
function reading(f, F::ApxCholFactor)
    ptr = lock(F.state) do
        isnull(F.ptr) && throw(closed_error(F))
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

function native_stats(ptr::Union{SolverPtr,FactorPtr})
    backend, n, factor_nnz, lumped_offdiag, rounds, peak_graph_bytes, setup_seconds, sddm,
        setup_max_threads, storage = @native ptr isa SolverPtr ? Native.solver_stats(ptr) :
                                                                 Native.factor_stats(ptr)
    return (backend=Int(backend), n=Int(n), factor_nnz=Int(factor_nnz),
            lumped_offdiag=Int(lumped_offdiag), rounds=Int(rounds),
            peak_graph_bytes=Int(peak_graph_bytes), setup_seconds=Float64(setup_seconds),
            sddm=sddm == 1, setup_max_threads=Int(setup_max_threads), factor_storage=Int(storage))
end

stats_tuple(s) =
    (backend=symbol(BACKENDS, s.backend), n=s.n, factor_nnz=s.factor_nnz,
     lumped_offdiag=s.lumped_offdiag, rounds=s.rounds, peak_graph_bytes=s.peak_graph_bytes,
     setup_seconds=s.setup_seconds, sddm=s.sddm, setup_max_threads=s.setup_max_threads,
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
        return stats_tuple(GC.@preserve S native_stats(ptr))
    finally
        unlock(S.lock)
    end
end
stats(F::ApxCholFactor) = stats_tuple(reading(native_stats, F))

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
            factor_storage=storage_type(o.factor_storage),
            threads=Int(o.threads), keep_factor=o.keep_factor_values == 1,
            factorization_tuple(o)...)
end
options(F::ApxCholFactor) = (threads=Int(F.options.threads), factorization_tuple(F.options)...)
factorization_tuple(o::Options) =
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

# perm[v] is v's elimination position; p = invperm(perm).
function export_arrays(H::Handle, ptr::Union{SolverPtr,FactorPtr}, values::Bool)
    nnz = values ? native_stats(ptr).factor_nnz : 0
    colptr = values ? Vector{Int64}(undef, H.n + 1) : Int64[]
    rowval = Vector{Int64}(undef, nnz)
    nzval = Vector{Float32}(undef, nnz)
    perm = Vector{Int64}(undef, H.n)
    export! = ptr isa SolverPtr ? Native.export_solver_factor : Native.export_factor
    GC.@preserve colptr rowval nzval perm @native export!(
        ptr, values ? pointer(colptr) : Ptr{Int64}(C_NULL), pointer(rowval), pointer(nzval),
        pointer(perm))
    p = invperm(perm)
    return values ? ExportedFactor(SparseMatrixCSC(H.n, H.n, colptr, rowval, nzval), p) : p
end

function export_from(H::ApxCholSolver, values::Bool)
    ptr = acquire(H)
    try
        return GC.@preserve H export_arrays(H, ptr, values)
    finally
        unlock(H.lock)
    end
end
export_from(H::ApxCholFactor, values::Bool) = reading(ptr -> export_arrays(H, ptr, values), H)

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
