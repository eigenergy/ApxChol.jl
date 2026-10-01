using ApxChol
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

@testset "ApxChol" begin
    @testset "C ABI layout and library" begin
        @test sizeof(ApxChol.COptions) == 64
        @test fieldoffset(ApxChol.COptions, 3) == 8           # tol
        @test fieldoffset(ApxChol.COptions, 12) == 48         # degree_quantile
        @test fieldoffset(ApxChol.COptions, 13) == 56         # exact_clique_max_degree
        @test sizeof(ApxChol.CSolveInfo) == 32
        @test fieldoffset(ApxChol.CSolveInfo, 3) == 8         # iterations
        @test sizeof(ApxChol.CStats) == 64
        @test fieldoffset(ApxChol.CStats, 9) == 56            # sddm
        @test occursin('+', ApxChol.version())
        @test ApxChol.openmp_enabled()
        @test ApxChol.backend_available(:cpu)
    end

    @testset "defaults" begin
        S = ApxCholSolver(grid_laplacian(4, 4))
        o = ApxChol.options(S)
        @test o.backend == :cpu && o.rtol == 1e-8 && o.maxiter == 200
        @test o.stagnation_window == 50 && o.seed == 42 && o.threads == 0
        @test o.sampler == :gks && o.partitioner == :block_greedy && o.storage == :vec_pool_aos
        @test !o.keep_factor
        close(S)
    end

    @testset "Laplacian: min-norm solution, threads $threads" for threads in
            unique((1, min(4, Sys.CPU_THREADS)))
        A = grid_laplacian(30, 30)
        b = compatible_rhs(size(A, 1))
        S = ApxCholSolver(A; rtol=1e-10, threads=threads, seed=7)
        r = solve(S, b)
        @test r.converged
        @test r.relative_residual < 1e-10
        @test relres(A, r.x, b) < 1e-8
        @test abs(sum(r.x)) < 1e-10 * norm(r.x, 1)
        @test r.iterations >= 1 && r.solve_seconds >= 0
        @test !ApxChol.stats(S).sddm
        @test ApxChol.stats(S).setup_max_threads == threads
    end

    @testset "SDDM agrees with a direct solve" begin
        A = sparse([3.0 -1.0; -1.0 2.0])
        b = [1.0, 4.0]
        r = solve(A, b; rtol=1e-12, threads=2)
        @test r.converged
        @test r.x ≈ A \ b rtol = 1e-10
        @test ApxCholSolver(A) \ b ≈ A \ b rtol = 1e-7
    end

    @testset "repeated solves, warm start, solve!, aliasing-free buffers" begin
        A = grid_laplacian(20, 20; shift=0.01)
        b = collect(range(-1.0, 3.0; length=size(A, 1)))
        S = ApxCholSolver(A)
        r1 = solve(S, b); r2 = solve(S, b)
        @test r1.x == r2.x && r1.iterations == r2.iterations
        w = solve(S, b; x0=r1.x, rtol=1e-6)
        @test w.converged && w.iterations == 0
        x = zeros(size(A, 1))
        r3 = solve!(x, S, b)
        @test r3.x === x && x == r1.x
        @test solve(S, Float32.(b)).converged
    end

    @testset "block solves equal column solves" begin
        A = grid_laplacian(12, 12)
        n = size(A, 1)
        B = hcat(compatible_rhs(n), zeros(n), compatible_rhs(n; salt=5))
        S = ApxCholSolver(A)
        R = solve(S, B)
        @test R isa BlockSolveResult && all(R.converged)
        for k in 1:3
            r = solve(S, B[:, k])
            @test R.X[:, k] == r.x
            @test R.iterations[k] == r.iterations
            @test R.relative_residuals[k] == r.relative_residual
        end
        @test R.iterations[2] == 0 && iszero(R.X[:, 2])
        @test S \ B == R.X
        @test solve(A, B).X ≈ R.X
    end

    @testset "non-convergence is reported, not thrown" begin
        A = grid_laplacian(5, 5; shift=0.05)
        b = collect(range(1.0, 2.0; length=size(A, 1)))
        S = ApxCholSolver(A; stagnation_window=0)
        r = solve(S, b; rtol=1e-300, maxiter=1)
        @test !r.converged
        @test r.iterations == 1
        @test r.relative_residual > 0
        @test norm(r.x) > 0
        S1 = ApxCholSolver(A; rtol=1e-300, maxiter=1)
        @test_throws NotConvergedError S1 \ b
        @test_throws NotConvergedError ldiv!(zeros(size(A, 1)), S1, b)
        R = solve(S1, hcat(b, b))
        @test !any(R.converged)
    end

    @testset "argument and operator errors" begin
        @test_throws DimensionMismatch ApxCholSolver(spzeros(2, 3))
        S = ApxCholSolver(grid_laplacian(3, 3))
        @test_throws DimensionMismatch solve(S, zeros(4))
        @test_throws DimensionMismatch solve(S, zeros(9); x0=zeros(2))
        @test_throws ArgumentError ApxCholSolver(grid_laplacian(3, 3); threads=-1)
        @test_throws ArgumentError ApxCholSolver(grid_laplacian(3, 3); sampler=:nope)
        @test_throws ArgumentError ApxCholSolver(grid_laplacian(3, 3); seed=-1)
        e = try solve(S, fill(NaN, 9)); nothing catch err; err end
        @test e isa ApxCholError && e.status == :invalid_argument
        e = try ApxCholSolver(grid_laplacian(3, 3); rtol=0.0); nothing catch err; err end
        @test e isa ApxCholError && e.status == :invalid_argument
        adjacency = sparse([0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0])
        e = try ApxCholSolver(adjacency); nothing catch err; err end
        @test e isa ApxCholError && e.status == :invalid_operator
        @test occursin("positive diagonal", e.message)
        @test occursin("ApxCholError(:invalid_operator)", sprint(showerror, e))
    end

    @testset "lumping of positive off-diagonals" begin
        A = grid_laplacian(4, 4; shift=0.5)
        A[2, 1] = -A[2, 1]; A[1, 2] = -A[1, 2]
        S = ApxCholSolver(A)
        s = ApxChol.stats(S)
        @test s.lumped_offdiag == 2
        @test s.sddm
        b = ones(size(A, 1))
        r = solve(S, b; rtol=1e-10)
        @test r.converged && relres(A, r.x, b) < 1e-8
    end

    @testset "factor export" begin
        A = grid_laplacian(9, 7)
        n = size(A, 1)
        S = ApxCholSolver(A)
        perm = ApxChol.permutation(S)
        @test sort(perm) == 1:n
        e = try ApxChol.factor(S); nothing catch err; err end
        @test e isa ApxCholError && e.status == :no_factor_values
        K = ApxCholSolver(A; keep_factor=true)
        L, p = ApxChol.factor(K)
        @test p == ApxChol.permutation(K)
        @test istril(L) && size(L) == (n, n)
        @test nnz(L) == ApxChol.stats(K).factor_nnz
        @test all(L.rowval[L.colptr[j]] == j for j in 1:n)       # diagonal first
    end

    @testset "preconditioner for Krylov.jl" begin
        A = grid_laplacian(40, 40; shift=1e-3)
        b = compatible_rhs(size(A, 1)) .+ 0.5     # not an eigenvector of A
        S = ApxCholSolver(A)
        P = ApxChol.aspreconditioner(S)
        @test size(P) == size(A)
        z = P \ b
        y = zeros(length(b)); ldiv!(y, P, b)
        @test y == z
        w = copy(b); ldiv!(P, w)
        @test w == z
        x, stats = Krylov.cg(A, b; M=P, ldiv=true, rtol=1e-10, atol=0.0)
        @test stats.solved
        @test relres(A, x, b) < 1e-8          # Krylov stops on the M-norm residual
        _, plain = Krylov.cg(A, b; rtol=1e-10, atol=0.0)
        @test stats.niter < plain.niter
    end

    @testset "lifetime: finalizers and close" begin
        A = grid_laplacian(6, 6; shift=0.1)
        for _ in 1:50
            ApxCholSolver(A)
        end
        GC.gc(); GC.gc()
        S = ApxCholSolver(A)
        @test isopen(S)
        close(S); close(S)
        @test !isopen(S)
        @test_throws ArgumentError solve(S, ones(size(A, 1)))
        @test occursin("closed", sprint(show, S))
    end

    @testset "concurrent solvers on Julia threads" begin
        A = grid_laplacian(16, 16)
        results = Vector{Any}(undef, 4)
        Threads.@threads for t in 1:4
            S = ApxCholSolver(A; threads=1)
            results[t] = solve(S, compatible_rhs(size(A, 1); salt=t))
            close(S)
        end
        @test all(r -> r.converged, results)
    end

    @testset "one OpenMP runtime; thread limits are call-scoped" begin
        @test haskey(ENV, "KMP_GTID_MODE")                  # set by __init__ unless preset
        before = omp_max_threads()
        omp_set_threads(3)
        @test ApxChol.max_threads() == 3                     # the same runtime instance
        S = ApxCholSolver(grid_laplacian(70, 70); threads=2)
        @test ApxChol.stats(S).setup_max_threads == 2
        @test omp_max_threads() == 3
        @test solve(S, compatible_rhs(4900)).converged
        @test omp_max_threads() == 3
        omp_set_threads(before)
        # Julia's own runtime may load its private libgomp (Linux, Julia 1.13);
        # every other OpenMP runtime in the process must be the JLL's libomp.
        julia_private = realpath(joinpath(Sys.BINDIR, Base.PRIVATE_LIBDIR))
        runtimes = unique(realpath.(filter(Libdl.dllist()) do path
            name = lowercase(basename(path))
            any(prefix -> startswith(name, prefix), ("libomp", "libiomp", "libgomp")) &&
                !startswith(realpath(path), julia_private)
        end))
        @test runtimes == [realpath(LLVMOpenMP_jll.libomp_path)]
    end

    @testset "backends" begin
        if ApxChol.backend_available(:metal)
            @info "Metal backend available; its tests live in test/metal.jl"
        else
            e = try ApxCholSolver(grid_laplacian(3, 3); backend=:metal); nothing catch err; err end
            @test e isa ApxCholError && e.status == :unsupported
        end
        @test_throws ArgumentError ApxCholSolver(grid_laplacian(3, 3); backend=:cuda)
    end
end
