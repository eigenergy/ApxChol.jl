# Metal backend (Apple silicon builds with a usable GPU), included by runtests.jl.

@testset "Metal backend" begin
    @testset "Laplacian converges on the original operator; agrees with the CPU" begin
        A = grid_laplacian(40, 40)
        b = compatible_rhs(size(A, 1))
        G = ApxCholSolver(A; backend=:metal, rtol=1e-10)
        C = ApxCholSolver(A; rtol=1e-10)
        @test ApxChol.options(G).backend == :metal && ApxChol.stats(G).backend == :metal
        @test G.backend == :metal && !ApxChol.stats(G).sddm
        g = ApxChol.solve(G, b); c = ApxChol.solve(C, b)
        @test g.converged && c.converged && g.relative_residual < 1e-10
        @test relres(A, g.x, b) ≈ g.relative_residual rtol = 1e-6
        @test abs(sum(g.x)) < 1e-10 * norm(g.x, 1)
        @test norm(g.x - c.x) / norm(c.x) < 1e-6
    end

    @testset "AUTO chooses Metal above 32,768 rows, the CPU for out-of-range operators" begin
        A = grid_laplacian(182, 182)                     # 33,124 rows
        b = compatible_rhs(size(A, 1))
        S = ApxCholSolver(A)
        @test S.backend == :metal && ApxChol.stats(S).backend == :metal
        @test ApxChol.solve(S, b).converged
        @test ApxCholSolver(A, ApxCholFactor(A)).backend == :cpu   # adoption under AUTO
        A[1, 2] = A[2, 1] = -1e-35                       # below Metal's 2^-100 floor
        @test ApxCholSolver(A).backend == :cpu
    end

    @testset "SDDM with a non-fp32-exact operator; extreme scales" begin
        A = grid_laplacian(25, 25; shift=0.1) * (1 + 1e-9)
        b = collect(range(-1.0, 2.0; length=size(A, 1)))
        G = ApxCholSolver(A; backend=:metal, rtol=1e-10)
        r = ApxChol.solve(G, b)
        @test r.converged && relres(A, r.x, b) < 1e-10
        @test r.x ≈ A \ b rtol = 1e-8
        for e in (-700, 700)
            s = ApxChol.solve(G, ldexp.(b, e))
            @test s.converged && norm(ldexp.(s.x, -e) - r.x) <= 1e-9 * norm(r.x)
        end
    end

    @testset "block solves equal single solves" begin
        A = grid_laplacian(20, 20)
        n = size(A, 1)
        B = hcat(compatible_rhs(n), zeros(n), 1e6 .* compatible_rhs(n; salt=3),
                 1e-12 .* compatible_rhs(n; salt=5))
        G = ApxCholSolver(A; backend=:metal)
        R = ApxChol.solve(G, B)
        @test all(R.converged) && R.iterations[2] == 0 && iszero(R.X[:, 2])
        for k in axes(B, 2)
            r = ApxChol.solve(G, B[:, k])
            @test bits(R.X[:, k]) == bits(r.x) && R.iterations[k] == r.iterations
        end
    end

    @testset "preconditioner application agrees with the CPU's" begin
        A = grid_laplacian(30, 30; shift=0.05)
        r = compatible_rhs(size(A, 1)) .+ 0.3
        F = ApxCholSolver(A; seed=11, threads=1)
        G = ApxCholSolver(A; seed=11, threads=1, backend=:metal)
        @test ApxChol.ordering(F) == ApxChol.ordering(G)    # same host factorization
        zf = ApxChol.aspreconditioner(F) \ r
        zg = ApxChol.aspreconditioner(G) \ r
        @test norm(zg - zf) / norm(zf) < 1e-5
    end

    @testset "factor reuse: bit identity on the own matrix, a nearby operator" begin
        for shift in (0.0, 0.05)
            A = grid_laplacian(30, 30; shift)
            n = size(A, 1)
            F = ApxCholFactor(A; seed=11, threads=1)
            G = ApxCholSolver(A; backend=:metal, seed=11, threads=1, keep_factor=true)
            H = ApxCholSolver(A, F; backend=:metal)
            @test ApxChol.stats(H).backend == :metal
            @test ApxChol.factor(ApxCholFactor(G)) == ApxChol.factor(F)
            B = hcat(compatible_rhs(n), compatible_rhs(n; salt=3) .+ shift, zeros(n))
            RG, RH = ApxChol.solve(G, B), ApxChol.solve(H, B)
            @test all(RH.converged) && bits(RG.X) == bits(RH.X) && RG.iterations == RH.iterations
            A2 = reweighted(A; shift)
            S = ApxCholSolver(A2, F; backend=:metal, rtol=1e-10, maxiter=1000)
            b = compatible_rhs(n; salt=4)
            r = ApxChol.solve(S, b)
            @test r.converged && relres(A2, r.x, b) < 1e-10
            @test relres(A2, r.x, b) ≈ r.relative_residual rtol = 1e-6
            foreach(close, (F, G, H, S))
        end
        A = grid_laplacian(12, 15)
        F = ApxCholFactor(ApxCholSolver(A; backend=:metal, keep_factor=true))
        @test ApxChol.stats(F).backend == :cpu
        @test ApxChol.solve(ApxCholSolver(A, F), compatible_rhs(size(A, 1); salt=6)).converged
        e = caught(() -> ApxCholSolver(grid_laplacian(12, 15; shift=0.1), F; backend=:metal))
        @test e isa ApxCholError && e.status == :invalid_argument
    end

    @testset "factor storage and honest non-convergence" begin
        A = grid_laplacian(10, 10; shift=0.05)
        b = ones(size(A, 1))
        H = ApxCholSolver(A; backend=:metal, factor_storage=Float16)
        @test ApxChol.stats(H).factor_storage == Float16 && ApxChol.solve(H, b).converged
        G = ApxCholSolver(A; backend=:metal, stagnation_window=0)
        r = ApxChol.solve(G, b; rtol=1e-30, maxiter=3)
        @test !r.converged && r.iterations == 3
        @test_throws NotConvergedError ApxCholSolver(A; backend=:metal, rtol=1e-30, maxiter=3) \ b
        for _ in 1:20
            close(ApxCholSolver(A; backend=:metal))
        end
        GC.gc()
        @test ApxChol.solve(G, b).converged
    end
end
