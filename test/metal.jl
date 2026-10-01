# Metal backend (Apple silicon builds with a usable GPU). Included by runtests.jl.

@testset "Metal backend" begin
    @testset "Laplacian converges on the original operator; agrees with the CPU" begin
        A = grid_laplacian(40, 40)
        b = compatible_rhs(size(A, 1))
        G = ApxCholSolver(A; backend=:metal, rtol=1e-10)
        C = ApxCholSolver(A; rtol=1e-10)
        @test ApxChol.options(G).backend == :metal
        @test ApxChol.stats(G).backend == :metal
        @test !ApxChol.stats(G).sddm
        g = solve(G, b); c = solve(C, b)
        @test g.converged && c.converged
        @test g.relative_residual < 1e-10
        @test relres(A, g.x, b) < 1e-10        # Metal reports the fp64 original residual
        @test relres(A, g.x, b) ≈ g.relative_residual rtol = 1e-6
        @test abs(sum(g.x)) < 1e-10 * norm(g.x, 1)
        # Both solve A x = b to 1e-10, so they agree to within the conditioning.
        @test norm(g.x - c.x) / norm(c.x) < 1e-6
    end

    @testset "SDDM with a non-fp32-exact operator" begin
        A = grid_laplacian(25, 25; shift=0.1) * (1 + 1e-9)
        b = collect(range(-1.0, 2.0; length=size(A, 1)))
        r = solve(A, b; backend=:metal, rtol=1e-10)
        @test r.converged
        @test relres(A, r.x, b) < 1e-10
        @test r.x ≈ A \ b rtol = 1e-8
    end

    @testset "lockstep block solves: per-column results equal single solves" begin
        A = grid_laplacian(20, 20)
        n = size(A, 1)
        B = hcat(compatible_rhs(n), zeros(n), 1e6 .* compatible_rhs(n; salt=3),
                 1e-12 .* compatible_rhs(n; salt=5))
        G = ApxCholSolver(A; backend=:metal)
        R = solve(G, B)
        @test all(R.converged)
        @test R.iterations[2] == 0 && iszero(R.X[:, 2])
        for k in axes(B, 2)
            r = solve(G, B[:, k])
            @test R.X[:, k] == r.x
            @test R.iterations[k] == r.iterations
            k == 2 || @test relres(A, R.X[:, k], B[:, k]) < 1e-8
        end
        W0 = reduce(hcat, (compatible_rhs(n; salt=s) for s in 1:70))  # wider than one batch
        W = solve(G, W0)
        @test all(W.converged)
        @test W.X[:, 64] == solve(G, W0[:, 64]).x
    end

    @testset "preconditioner application agrees with the CPU's" begin
        A = grid_laplacian(30, 30; shift=0.05)
        r = compatible_rhs(size(A, 1)) .+ 0.3
        F = ApxCholSolver(A; seed=11, threads=1)
        G = ApxCholSolver(A; seed=11, threads=1, backend=:metal)
        @test ApxChol.permutation(F) == ApxChol.permutation(G)   # same host factorization
        zf = ApxChol.aspreconditioner(F) \ r
        zg = ApxChol.aspreconditioner(G) \ r
        @test norm(zg - zf) / norm(zf) < 1e-5                   # fp32 triangular solves
    end

    @testset "honest non-convergence and lifetime" begin
        A = grid_laplacian(10, 10; shift=0.05)
        b = ones(size(A, 1))
        G = ApxCholSolver(A; backend=:metal, stagnation_window=0)
        r = solve(G, b; rtol=1e-30, maxiter=3)
        @test !r.converged && r.iterations == 3
        @test_throws NotConvergedError ApxCholSolver(A; backend=:metal, rtol=1e-30, maxiter=3) \ b
        for _ in 1:20
            close(ApxCholSolver(A; backend=:metal))
        end
        GC.gc()
        @test solve(G, b).converged
    end
end
