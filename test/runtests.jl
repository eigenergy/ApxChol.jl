using ApxChol
using ApxChol: ApxCholPreconditioner, BlockSolveResult, ExportedFactor, SolveResult
using Aqua
using Krylov
using LLVMOpenMP_jll
using Libdl
using LinearAlgebra
using SparseArrays
using Test

const omp_max_threads() = ccall((:omp_get_max_threads, LLVMOpenMP_jll.libomp_path), Cint, ())
const omp_set_threads(n) = ccall((:omp_set_num_threads, LLVMOpenMP_jll.libomp_path), Cvoid, (Cint,), n)

"Weighted grid Laplacian, plus `shift` on the diagonal (SDDM when positive)."
function grid_laplacian(rows, cols; shift=0.0)
    id(r, c) = (r - 1) * cols + c
    I = Int[]; J = Int[]; V = Float64[]
    edge(u, v, w) = (append!(I, (u, v)); append!(J, (v, u)); append!(V, (-w, -w)))
    for r in 1:rows, c in 1:cols
        w = 1.0 + 0.25 * ((7r + 3c) % 5)
        r < rows && edge(id(r, c), id(r + 1, c), w)
        c < cols && edge(id(r, c), id(r, c + 1), w)
    end
    n = rows * cols
    A = sparse(I, J, V, n, n)
    return A - spdiagm(0 => vec(sum(A; dims=2)) .- shift)
end

compatible_rhs(n; salt=1) = (b = [sin(0.37 * i * salt) + 0.1salt for i in 1:n]; b .- sum(b) / n)
relres(A, x, b) = norm(A * x - b) / norm(b)
bits(x) = reinterpret(UInt64, x)
# Two correct evaluations of ‖b − A x‖/‖b‖ differ by the rounding of forming b − A x.
residual_rounding(A, x, b) = 32eps() * norm(abs.(A) * abs.(x)) / norm(b)

"`A` with its off-diagonals scaled by symmetric factors in [0.6, 1.8] and `shift` on the diagonal."
function reweighted(A; shift=0.0)
    I, J, V = findnz(A)
    off = findall(I .!= J)
    W = [V[k] * (0.6 + 0.3 * ((5min(I[k], J[k]) + 3max(I[k], J[k])) % 5)) for k in off]
    B = sparse(I[off], J[off], W, size(A)...)
    return B - spdiagm(0 => vec(sum(B; dims=2)) .- shift)
end

"""`count` cliques of `size` vertices in a light ring: the first round picks one vertex per
clique, a yield low enough that the factorization hands the rest over unless the selection
reaches `omp_threshold`."""
function clique_ring(count, size)
    I = Int[]; J = Int[]; V = Float64[]
    edge(u, v, w) = (append!(I, (u, v)); append!(J, (v, u)); append!(V, (-w, -w)))
    for c in 0:count-1
        base = c * size
        for i in 0:size-1, j in i+1:size-1
            edge(base + i + 1, base + j + 1, 1.0 + 0.125 * ((i + j) % 4))
        end
        edge(base + 1, mod(c + 1, count) * size + 2, 0.5)
    end
    A = sparse(I, J, V, count * size, count * size)
    return A - spdiagm(0 => vec(sum(A; dims=2)))
end

"The exception `f()` throws, or `nothing`."
caught(f) = try f(); nothing catch e; e end

@testset "ApxChol" begin
    include("abi.jl")
    include("solve.jl")
    include("factor.jl")
    include("safety.jl")
    include("boundary.jl")
    if ApxChol.backend_available(:metal)
        include("metal.jl")
    else
        @info "Metal unavailable in this build or on this machine; skipping test/metal.jl"
    end
    include("digest.jl")
    include("aqua.jl")
end
