# Solves, LinearAlgebra integration and the preconditioner.

"""
    ApxChol.SolveResult

One solve: the solution `x`, PCG `iterations`, the original system's
`relative_residual` ‖b − A x‖/‖b‖ of `x` (recomputed in FP64 by apxchol on
every backend), `converged` (`relative_residual < rtol`) and `solve_seconds`.
A solve that does not converge still returns its final iterate.
"""
struct SolveResult{V<:AbstractVector{Float64}}
    x::V
    iterations::Int
    relative_residual::Float64
    converged::Bool
    solve_seconds::Float64
end

"""
    ApxChol.BlockSolveResult

A solve with several right-hand sides: the solutions `X`, one column per
right-hand side, and per column `iterations`, `relative_residuals` and
`converged`.
"""
struct BlockSolveResult{M<:AbstractMatrix{Float64}}
    X::M
    iterations::Vector{Int}
    relative_residuals::Vector{Float64}
    converged::Vector{Bool}
end

"""
    NotConvergedError(result)

Thrown by `\\` and `ldiv!`, which have no convergence flag to return, when a
solve did not reach its tolerance. `result` holds the final iterate.
"""
struct NotConvergedError <: Exception
    result::Union{SolveResult,BlockSolveResult}
end
function Base.showerror(io::IO, e::NotConvergedError)
    r = e.result
    if r isa SolveResult
        print(io, "NotConvergedError: relative residual ", r.relative_residual, " after ",
              r.iterations, " iterations")
    else
        print(io, "NotConvergedError: ", count(!, r.converged), " of ", length(r.converged),
              " columns did not converge")
    end
end

# ── Memory apxchol reads and writes in place ─────────────────────────────────

# Dense Float64 storage apxchol can address directly: unit stride, and for a
# matrix column-major without gaps (contiguous views included).
native(v::AbstractVector) = v isa StridedVector{Float64} && stride(v, 1) == 1
native(M::AbstractMatrix) =
    M isa StridedMatrix{Float64} && stride(M, 1) == 1 && stride(M, 2) == size(M, 1)

# Exactly aliased arrays are allowed (apxchol copies them first); partially
# overlapping ones are undefined in C.
function partial_overlap(a, b)
    pa, pb = UInt(pointer(a)), UInt(pointer(b))
    return pa != pb && pa < pb + sizeof(b) && pb < pa + sizeof(a)
end

# Whether apxchol can read `v` in place while it writes `out`.
readable(::Nothing, out) = true
readable(v, out) = native(v) && !partial_overlap(v, out)

# The general path's copies, which apxchol can always address.
dense(::Nothing) = nothing
dense(v::AbstractVector) = Vector{Float64}(v)
dense(M::AbstractMatrix) = Matrix{Float64}(M)

pointer_or_null(::Nothing) = Ptr{Float64}(C_NULL)
pointer_or_null(v) = pointer(v)

# The array that owns the memory of `v` (a view's parent): what a native call
# keeps rooted. Preserving the owner rather than the view avoids boxing it.
owner(v::AbstractArray) = (p = parent(v); p === v ? v : owner(p))
owner(::Nothing) = nothing

call_tol(rtol) = rtol === nothing ? -1.0 : Float64(rtol)
call_maxiter(maxiter) = maxiter === nothing ? Int32(-1) :
    maxiter >= 0 ? Int32(maxiter) : throw(ArgumentError("maxiter must be non-negative"))

function check_length(v, n, what)
    length(v) == n || throw(DimensionMismatch("$what has length $(length(v)), expected $n"))
    return nothing
end

# ── Solves ───────────────────────────────────────────────────────────────────

# One apxchol_solver_solve on arrays apxchol can address; returns its info.
function native_solve!(S::ApxCholSolver, out, rhs, guess, tol::Float64, iters::Int32)
    ptr = acquire(S)
    try
        out_owner, rhs_owner, guess_owner = owner(out), owner(rhs), owner(guess)
        status = GC.@preserve S out_owner rhs_owner guess_owner @native libapxchol.apxchol_solver_solve(
            ptr::Ptr{Cvoid}, pointer(rhs)::Ptr{Float64}, pointer_or_null(guess)::Ptr{Float64},
            pointer(out)::Ptr{Float64}, tol::Float64, iters::Int32, S.info::Ref{CSolveInfo},
            S.error_buffer::Ptr{UInt8}, ERROR_CAPACITY::Csize_t)::Int32
        check(status, S.error_buffer; not_converged_ok=true)
        return S.info[]
    finally
        unlock(S.lock)
    end
end

"""
    ApxChol.solve!(x, S::ApxCholSolver, b; x0=nothing, rtol=nothing, maxiter=nothing) -> SolveResult
    ApxChol.solve!(X, S::ApxCholSolver, B; X0=nothing, rtol=nothing, maxiter=nothing) -> BlockSolveResult

Solve `A x = b` (every column of `A X = B`) into `x` (`X`), optionally from the
initial guess `x0` (`X0`). `rtol` and `maxiter` default to the solver's.
Non-convergence is reported in the result, not thrown. `x` may be `b` itself.
apxchol rescales extreme inputs internally: a `b` and `x0` that cannot share
one power-of-two scale throw `ApxCholError(:invalid_argument)`, and a solution
outside the finite range throws `ApxCholError(:unsupported)` and leaves `x`
unspecified.

Contiguous `Float64` arrays, views included, are passed to apxchol in place:
a warmed-up `solve!` on them allocates nothing. Other arrays are converted.
"""
function solve!(x::AbstractVector{Float64}, S::ApxCholSolver, b::AbstractVector{<:Real};
                x0::Union{Nothing,AbstractVector{<:Real}}=nothing, rtol=nothing, maxiter=nothing)
    check_length(b, S.n, "b"); check_length(x, S.n, "x")
    x0 === nothing || check_length(x0, S.n, "x0")
    tol, iters = call_tol(rtol), call_maxiter(maxiter)
    info = if native(x) && readable(b, x) && readable(x0, x)
        native_solve!(S, x, b, x0, tol, iters)
    else
        out = Vector{Float64}(undef, S.n)
        i = native_solve!(S, out, dense(b), dense(x0), tol, iters)
        copyto!(x, out)
        i
    end
    return SolveResult(x, Int(info.iterations), info.relative_residual, info.converged == 1,
                       info.solve_seconds)
end

function native_solve_block!(S::ApxCholSolver, out, rhs, guess, tol::Float64, iters::Int32,
                             iterations, residuals, converged)
    ptr = acquire(S)
    try
        out_owner, rhs_owner, guess_owner = owner(out), owner(rhs), owner(guess)
        status = GC.@preserve S out_owner rhs_owner guess_owner @native libapxchol.apxchol_solver_solve_block(
            ptr::Ptr{Cvoid}, size(out, 2)::Int64, pointer(rhs)::Ptr{Float64},
            pointer_or_null(guess)::Ptr{Float64}, pointer(out)::Ptr{Float64}, tol::Float64,
            iters::Int32, iterations::Ptr{Int64}, residuals::Ptr{Float64},
            converged::Ptr{Int32}, S.error_buffer::Ptr{UInt8}, ERROR_CAPACITY::Csize_t)::Int32
        check(status, S.error_buffer; not_converged_ok=true)
    finally
        unlock(S.lock)
    end
    return nothing
end

function solve!(X::AbstractMatrix{Float64}, S::ApxCholSolver, B::AbstractMatrix{<:Real};
                X0::Union{Nothing,AbstractMatrix{<:Real}}=nothing, rtol=nothing, maxiter=nothing)
    size(B, 1) == S.n || throw(DimensionMismatch("B has $(size(B, 1)) rows, expected $(S.n)"))
    size(X) == size(B) || throw(DimensionMismatch("X has size $(size(X)), expected $(size(B))"))
    X0 === nothing || size(X0) == size(B) ||
        throw(DimensionMismatch("X0 has size $(size(X0)), expected $(size(B))"))
    k = size(B, 2)
    tol, iters = call_tol(rtol), call_maxiter(maxiter)
    iterations = Vector{Int64}(undef, k)
    residuals = Vector{Float64}(undef, k)
    converged = Vector{Int32}(undef, k)
    if native(X) && readable(B, X) && readable(X0, X)
        native_solve_block!(S, X, B, X0, tol, iters, iterations, residuals, converged)
    else
        out = Matrix{Float64}(undef, size(X))
        native_solve_block!(S, out, dense(B), dense(X0), tol, iters, iterations, residuals,
                            converged)
        copyto!(X, out)
    end
    return BlockSolveResult(X, Vector{Int}(iterations), residuals, Vector{Bool}(converged .== 1))
end

"""
    ApxChol.solve(S::ApxCholSolver, b; x0, rtol, maxiter) -> SolveResult
    ApxChol.solve(S::ApxCholSolver, B::AbstractMatrix; X0, rtol, maxiter) -> BlockSolveResult
    ApxChol.solve(A, b; x0=nothing, kwargs...) -> SolveResult or BlockSolveResult

Solve with an existing solver, or factorize `A` for one solve (the keywords of
[`ApxCholSolver`](@ref) apply). Non-convergence is reported, not thrown.
"""
solve(S::ApxCholSolver, b::AbstractVector{<:Real}; kwargs...) =
    solve!(Vector{Float64}(undef, S.n), S, b; kwargs...)
solve(S::ApxCholSolver, B::AbstractMatrix{<:Real}; kwargs...) =
    solve!(Matrix{Float64}(undef, size(B)), S, B; kwargs...)
function solve(A::Operator, b::AbstractVecOrMat{<:Real}; x0=nothing, kwargs...)
    S = ApxCholSolver(A; kwargs...)
    try
        return b isa AbstractVector ? solve(S, b; x0) : solve(S, b; X0=x0)
    finally
        close(S)
    end
end

function converged_or_throw(result)
    (result isa SolveResult ? result.converged : all(result.converged)) ||
        throw(NotConvergedError(result))
    return result
end

Base.:\(S::ApxCholSolver, b::AbstractVector{<:Real}) = converged_or_throw(solve(S, b)).x
Base.:\(S::ApxCholSolver, B::AbstractMatrix{<:Real}) = converged_or_throw(solve(S, B)).X
function LinearAlgebra.ldiv!(x::AbstractVector{Float64}, S::ApxCholSolver, b::AbstractVector{<:Real})
    converged_or_throw(solve!(x, S, b))
    return x
end
function LinearAlgebra.ldiv!(X::AbstractMatrix{Float64}, S::ApxCholSolver, B::AbstractMatrix{<:Real})
    converged_or_throw(solve!(X, S, B))
    return X
end
LinearAlgebra.ldiv!(S::ApxCholSolver, b::AbstractVector{Float64}) = ldiv!(b, S, b)
LinearAlgebra.ldiv!(S::ApxCholSolver, B::AbstractMatrix{Float64}) = ldiv!(B, S, B)

# ── Preconditioner ───────────────────────────────────────────────────────────

"""
    ApxChol.apply!(z, S::ApxCholSolver, r) -> z

One preconditioner application `z = M⁻¹ r`: the forward and backward
triangular solves with the approximate factor. `z` may be `r` itself; on
contiguous `Float64` vectors a warmed-up call allocates nothing.

With an SDDM factor (`stats(S).sddm`) this is a fixed linear map. With a
Laplacian factor `z` is defined only up to a constant vector: Metal centres
input and output on every application; the CPU route centres them on every
K-th application since the last solve (apxchol's center-k schedule,
`APXCHOL_CENTER_K`, default 10), so repeated applications to a mean-zero `r`
agree up to a constant vector.
"""
function apply!(z::AbstractVector{Float64}, S::ApxCholSolver, r::AbstractVector{<:Real})
    check_length(r, S.n, "r"); check_length(z, S.n, "z")
    if native(z) && readable(r, z)
        native_apply!(S, z, r)
    else
        out = Vector{Float64}(undef, S.n)
        native_apply!(S, out, dense(r))
        copyto!(z, out)
    end
    return z
end

function native_apply!(S::ApxCholSolver, out, r)
    ptr = acquire(S)
    try
        out_owner, r_owner = owner(out), owner(r)
        status = GC.@preserve S out_owner r_owner @native libapxchol.apxchol_solver_apply(
            ptr::Ptr{Cvoid}, pointer(r)::Ptr{Float64}, pointer(out)::Ptr{Float64},
            S.error_buffer::Ptr{UInt8}, ERROR_CAPACITY::Csize_t)::Int32
        check(status, S.error_buffer)
    finally
        unlock(S.lock)
    end
    return nothing
end

"""
    ApxChol.ApxCholPreconditioner

The preconditioner `M ≈ A` of an [`ApxCholSolver`](@ref): `ldiv!(y, P, x)`,
`ldiv!(P, x)` and `P \\ x` apply `M⁻¹` once ([`ApxChol.apply!`](@ref)). Use it
as `Krylov.cg(A, b; M=P, ldiv=true)` or `IterativeSolvers.cg(A, b; Pl=P)`.
"""
struct ApxCholPreconditioner
    solver::ApxCholSolver
end

"""
    ApxChol.aspreconditioner(S::ApxCholSolver) -> ApxCholPreconditioner
"""
aspreconditioner(S::ApxCholSolver) = ApxCholPreconditioner(S)

Base.size(P::ApxCholPreconditioner, d...) = size(P.solver, d...)
Base.eltype(::Type{ApxCholPreconditioner}) = Float64
LinearAlgebra.ldiv!(y::AbstractVector{Float64}, P::ApxCholPreconditioner, x::AbstractVector{<:Real}) =
    apply!(y, P.solver, x)
LinearAlgebra.ldiv!(P::ApxCholPreconditioner, x::AbstractVector{Float64}) = apply!(x, P.solver, x)
Base.:\(P::ApxCholPreconditioner, x::AbstractVector{<:Real}) =
    apply!(Vector{Float64}(undef, length(x)), P.solver, x)
