@testset "Factors" begin
    @testset "export: L and p with A[p, p] ≈ L L'" begin
        A = grid_laplacian(9, 7; shift=0.05)
        n = size(A, 1)
        S = ApxCholSolver(A)
        p = ApxChol.ordering(S)
        @test sort(p) == 1:n
        e = caught(() -> ApxChol.factor(S))
        @test e isa ApxCholError && e.status == :no_factor_values
        K = ApxCholSolver(A; keep_factor=true)
        f = ApxChol.factor(K)
        @test f isa ExportedFactor && f.p == ApxChol.ordering(K)
        @test f.L isa SparseMatrixCSC{Float32,Int64} && istril(f.L) && size(f.L) == (n, n)
        @test nnz(f.L) == ApxChol.stats(K).factor_nnz
        @test all(f.L.rowval[f.L.colptr[j]] == j for j in 1:n)     # diagonal first
        # The approximation is spectral: L L' preconditions A[p, p] well, which the
        # inverse permutation (apxchol's perm) does not.
        M = Float64.(f.L) * Float64.(f.L)'
        spread(P) = (λ = eigvals(Symmetric(Matrix(P)), Symmetric(Matrix(M))); maximum(λ) / minimum(λ))
        @test spread(A[f.p, f.p]) < 10
        @test spread(A[f.p, f.p]) < spread(A[invperm(f.p), invperm(f.p)]) / 10
        @test_throws MethodError ((L, perm) = f)                    # 0.3's tuple is gone
    end

    @testset "factor handles: export, stats, bit identity on the own matrix" begin
        for shift in (0.0, 0.05)
            A = grid_laplacian(24, 19; shift)
            n = size(A, 1)
            F = ApxCholFactor(A; threads=1, seed=7)
            @test size(F) == (n, n) && isopen(F)
            S = ApxCholSolver(A; threads=1, seed=7, keep_factor=true)
            fF, fS = ApxChol.factor(F), ApxChol.factor(S)
            @test fF == fS && reinterpret(UInt32, fF.L.nzval) == reinterpret(UInt32, fS.L.nzval)
            @test ApxChol.ordering(F) == fF.p
            sf, ss = ApxChol.stats(F), ApxChol.stats(S)
            @test sf.backend == :cpu && sf.n == n && sf.sddm == (shift > 0)
            @test sf.factor_storage == Float32
            @test (sf.factor_nnz, sf.rounds, sf.lumped_offdiag) == (ss.factor_nnz, ss.rounds, ss.lumped_offdiag)
            @test sf.setup_max_threads == 1 && sf.setup_seconds >= 0
            @test ApxChol.options(F) == (threads=1, seed=7, sampler=:gks, partitioner=:block_greedy,
                                         degree_quantile=ApxChol.DEFAULTS[].degree_quantile,
                                         exact_clique_max_degree=0, omp_threshold=2000)
            # A solver adopting F solves bit for bit as one that factorized A itself.
            T = ApxCholSolver(A, F; threads=1, rtol=1e-10)
            U = ApxCholSolver(A; threads=1, seed=7, rtol=1e-10)
            @test ApxChol.stats(T).factor_nnz == sf.factor_nnz && T.backend == :cpu
            @test ApxChol.options(T).seed == 7 && ApxChol.options(T).rtol == 1e-10
            b = compatible_rhs(n) .+ shift
            for x0 in (nothing, 0.1 .* compatible_rhs(n; salt=3))
                rt, ru = ApxChol.solve(T, b; x0), ApxChol.solve(U, b; x0)
                @test rt.converged && bits(rt.x) == bits(ru.x)
                @test rt.iterations == ru.iterations && rt.relative_residual == ru.relative_residual
            end
            @test bits(ApxChol.aspreconditioner(T) \ b) == bits(ApxChol.aspreconditioner(U) \ b)
            foreach(close, (F, S, T, U))
        end
    end

    @testset "factor reuse on another operator of the same size" begin
        for shift in (0.0, 0.05)
            A = grid_laplacian(26, 20; shift)
            B = reweighted(A; shift)
            @test B != A && issymmetric(B)
            F = ApxCholFactor(A)
            S = ApxCholSolver(B, F; rtol=1e-10, maxiter=1000)
            @test ApxChol.stats(S).factor_nnz == ApxChol.stats(F).factor_nnz
            b = compatible_rhs(size(B, 1); salt=4)
            r = ApxChol.solve(S, b)
            @test r.converged && relres(B, r.x, b) < 1e-9
            shift == 0 && @test abs(sum(r.x)) < 1e-10 * norm(r.x, 1)
            own = ApxChol.solve(ApxCholSolver(B; rtol=1e-10, maxiter=1000), b)
            @test norm(r.x - own.x) / norm(own.x) < 1e-6
        end
        # An SDDM factor on a Laplacian: a solution, not the min-norm one.
        A = grid_laplacian(18, 18)
        S = ApxCholSolver(A, ApxCholFactor(grid_laplacian(18, 18; shift=0.02)); rtol=1e-10)
        b = compatible_rhs(size(A, 1); salt=2)
        r = ApxChol.solve(S, b)
        @test r.converged && relres(A, r.x, b) < 1e-9
    end

    @testset "factor handles: errors" begin
        A = grid_laplacian(6, 5)
        F = ApxCholFactor(A)
        @test_throws DimensionMismatch ApxCholSolver(grid_laplacian(5, 5), F)
        @test_throws MethodError ApxCholSolver(A, F; seed=1)          # no factorization keywords
        @test_throws InexactError ApxCholFactor(A; omp_threshold=-1)
        e = caught(() -> ApxCholSolver(grid_laplacian(6, 5; shift=0.1), F))
        @test e isa ApxCholError && e.status == :invalid_argument && occursin("Laplacian factor", e.message)
        adjacency = abs.(A - spdiagm(0 => diag(A)))
        for make in (() -> ApxCholSolver(adjacency, F), () -> ApxCholFactor(adjacency))
            e = caught(make)
            @test e isa ApxCholError && e.status == :invalid_operator
        end
        e = caught(() -> ApxCholFactor(ApxCholSolver(A)))
        @test e isa ApxCholError && e.status == :no_factor_values
        close(F); close(F)
        @test !isopen(F) && sprint(show, F) == "ApxCholFactor(30×30, closed)"
        for use in (() -> ApxCholSolver(A, F), () -> ApxChol.stats(F), () -> ApxChol.factor(F),
                    () -> ApxChol.ordering(F))
            e = caught(use)
            @test e isa InvalidStateException && e.state == :closed
        end
    end

    @testset "factor handles: a solver's factor round-trips" begin
        A = grid_laplacian(16, 21; shift=0.03)
        S = ApxCholSolver(A; threads=1, keep_factor=true, sampler=:trace_cycle)
        F = ApxCholFactor(S)
        @test ApxChol.stats(F).setup_seconds == ApxChol.stats(S).setup_seconds
        @test ApxChol.factor(F) == ApxChol.factor(S)
        @test ApxChol.options(F).sampler == :trace_cycle
        T = ApxCholSolver(A, F; threads=1, keep_factor=true)
        @test ApxChol.options(T).sampler == :trace_cycle            # the factor's options
        b = collect(range(-1.0, 2.0; length=size(A, 1)))
        @test bits(ApxChol.solve(T, b).x) == bits(ApxChol.solve(S, b).x)
        @test ApxChol.factor(ApxCholFactor(T)) == ApxChol.factor(S)
        # A factor carries no solve options: an adopting solver starts from the defaults.
        S3 = ApxCholSolver(A; rtol=1e-3, maxiter=7, seed=5, keep_factor=true)
        o = ApxChol.options(ApxCholSolver(A, ApxCholFactor(S3)))
        @test (o.rtol, o.maxiter, o.seed) == (1e-8, 200, 5)
        @test ApxChol.options(ApxCholSolver(A, ApxCholFactor(S3); rtol=1e-4)).rtol == 1e-4
    end

    @testset "omp_threshold reaches the factorization" begin
        A = clique_ring(40, 30)
        default = ApxChol.stats(ApxCholFactor(A; threads=1))
        one = ApxChol.stats(ApxCholFactor(A; threads=1, omp_threshold=1))
        @test one.rounds > default.rounds              # the low-yield first round is kept
        S = ApxCholSolver(A; threads=1, omp_threshold=1)
        @test ApxChol.options(S).omp_threshold == 1 && ApxChol.stats(S).rounds == one.rounds
    end
end
