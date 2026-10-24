@testset "Solves" begin
    @testset "defaults; AUTO resolves to the CPU for small inputs" begin
        S = ApxCholSolver(grid_laplacian(4, 4))
        o = ApxChol.options(S)
        @test o.backend == :auto && o.rtol == 1e-8 && o.maxiter == 200
        @test o.seed == 42 && o.threads == 0 && o.degree_quantile == -1
        @test o.sampler == :gks && o.partitioner == :block_greedy
        @test !o.keep_factor && o.omp_threshold == 2000 && o.factor_storage === nothing
        s = ApxChol.stats(S)
        @test s.backend == :cpu && S.backend == :cpu
        @test s.factor_storage == Float32
        @test sprint(show, S) == "ApxCholSolver(16×16, cpu)"
        close(S)
    end

    @testset "every option reaches apxchol and round-trips" begin
        A = grid_laplacian(14, 13; shift=0.01)
        kw = (backend=:cpu, rtol=1e-6, maxiter=50, seed=9, sampler=:trace_cycle,
              partitioner=:priority_greedy, degree_quantile=0.5, exact_clique_max_degree=3,
              omp_threshold=100, factor_storage=Float32, threads=2, keep_factor=true)
        S = ApxCholSolver(A; kw...)
        @test ApxChol.options(S) == (; backend=:cpu, rtol=1e-6, maxiter=50, factor_storage=Float32,
                                     threads=2, keep_factor=true, seed=9, sampler=:trace_cycle,
                                     partitioner=:priority_greedy, degree_quantile=0.5,
                                     exact_clique_max_degree=3, omp_threshold=100)
        @test ApxChol.stats(S).setup_max_threads == 2
        r = ApxChol.solve(S, ones(size(A, 1)))
        @test r.converged && r.relative_residual < 1e-6
        @test ApxChol.solve(ApxCholSolver(A; partitioner=:baumann_kyng), ones(size(A, 1))).converged
    end

    @testset "Laplacian: min-norm solution, threads $threads" for threads in
            unique((1, min(4, Sys.CPU_THREADS)))
        A = grid_laplacian(30, 30)
        b = compatible_rhs(size(A, 1))
        S = ApxCholSolver(A; rtol=1e-10, threads, seed=7)
        r = ApxChol.solve(S, b)
        @test r.converged && r.relative_residual < 1e-10
        @test abs(r.relative_residual - relres(A, r.x, b)) <= residual_rounding(A, r.x, b)
        @test abs(sum(r.x)) < 1e-10 * norm(r.x, 1)
        @test r.iterations >= 1 && r.solve_seconds >= 0
        @test !ApxChol.stats(S).sddm
        @test ApxChol.stats(S).setup_max_threads == threads
    end

    @testset "SDDM agrees with a direct solve" begin
        A = sparse([3.0 -1.0; -1.0 2.0])
        b = [1.0, 4.0]
        r = ApxChol.solve(A, b; rtol=1e-12, threads=2)
        @test r.converged
        @test r.x ≈ A \ b rtol = 1e-10
        @test ApxCholSolver(A) \ b ≈ A \ b rtol = 1e-7
    end

    @testset "input types: Int32 indices, Float32 values, symmetric wrappers" begin
        A = grid_laplacian(17, 11; shift=0.02)
        b = compatible_rhs(size(A, 1)) .+ 0.1
        reference = ApxChol.solve(ApxCholSolver(A; threads=1), b).x
        for M in (SparseMatrixCSC{Float64,Int32}(A), Symmetric(triu(A)), Symmetric(tril(A), :L),
                  Hermitian(triu(A)))
            @test bits(ApxChol.solve(ApxCholSolver(M; threads=1), b).x) == bits(reference)
        end
        A32 = SparseMatrixCSC{Float32,Int64}(A)        # different values: a different operator
        @test ApxChol.solve(ApxCholSolver(A32), b).converged
        @test_throws MethodError ApxCholSolver(complex.(A))
    end

    @testset "repeated solves, warm starts, views and in-place solves" begin
        A = grid_laplacian(20, 20; shift=0.01)
        n = size(A, 1)
        b = collect(range(-1.0, 3.0; length=n))
        S = ApxCholSolver(A; threads=1)
        r1 = ApxChol.solve(S, b); r2 = ApxChol.solve(S, b)
        @test bits(r1.x) == bits(r2.x) && r1.iterations == r2.iterations
        w = ApxChol.solve(S, b; x0=r1.x, rtol=1e-6)
        @test w.converged && w.iterations == 0
        x = zeros(n)
        @test ApxChol.solve!(x, S, b).x === x && bits(x) == bits(r1.x)
        @test ApxChol.solve(S, Float32.(b)).converged
        # Contiguous views go to apxchol in place; strided ones are converted.
        B = hcat(b, b); X = zeros(n, 3)
        r = ApxChol.solve!(view(X, :, 2), S, view(B, :, 1))
        @test r.x isa SubArray && bits(X[:, 2]) == bits(r1.x) && iszero(X[:, [1, 3]])
        long = repeat(b; inner=2)
        @test bits(ApxChol.solve(S, view(long, 1:2:2n)).x) == bits(r1.x)
        # In place: x is b itself; ldiv!(S, b) overwrites b.
        y = copy(b)
        @test bits(ApxChol.solve!(y, S, y).x) == bits(r1.x)
        y = copy(b); ldiv!(S, y)
        @test bits(y) == bits(r1.x)
    end

    @testset "block solves equal column solves; block warm starts" begin
        A = grid_laplacian(12, 12)
        n = size(A, 1)
        B = hcat(compatible_rhs(n), zeros(n), compatible_rhs(n; salt=5))
        S = ApxCholSolver(A)
        R = ApxChol.solve(S, B)
        @test R isa BlockSolveResult && all(R.converged)
        for k in 1:3
            r = ApxChol.solve(S, B[:, k])
            @test bits(R.X[:, k]) == bits(r.x)
            @test R.iterations[k] == r.iterations
            @test R.relative_residuals[k] == r.relative_residual
        end
        @test R.iterations[2] == 0 && iszero(R.X[:, 2])
        @test S \ B == R.X
        @test ApxChol.solve(A, B).X ≈ R.X
        W = ApxChol.solve(S, B; X0=R.X, rtol=1e-6)
        @test all(W.converged) && all(iszero, W.iterations)
        X = similar(B); ldiv!(X, S, B)
        @test X == R.X
        Y = copy(B); ldiv!(S, Y)
        @test Y == R.X
        @test size(ApxChol.solve(S, zeros(n, 0)).X) == (n, 0)
    end

    @testset "non-convergence is reported, not thrown" begin
        A = grid_laplacian(5, 5; shift=0.05)
        b = collect(range(1.0, 2.0; length=size(A, 1)))
        S = ApxCholSolver(A)
        r = ApxChol.solve(S, b; rtol=1e-300, maxiter=1)
        @test !r.converged && r.iterations == 1 && r.relative_residual > 0 && norm(r.x) > 0
        S1 = ApxCholSolver(A; rtol=1e-300, maxiter=1)
        e = caught(() -> S1 \ b)
        @test e isa NotConvergedError && occursin("after 1 iterations", sprint(showerror, e))
        @test_throws NotConvergedError ldiv!(zeros(size(A, 1)), S1, b)
        @test !any(ApxChol.solve(S1, hcat(b, b)).converged)
        @test_throws NotConvergedError S1 \ hcat(b, b)
    end

    @testset "argument and operator errors" begin
        @test_throws DimensionMismatch ApxCholSolver(spzeros(2, 3))
        S = ApxCholSolver(grid_laplacian(3, 3))
        @test_throws DimensionMismatch ApxChol.solve(S, zeros(4))
        @test_throws DimensionMismatch ApxChol.solve(S, zeros(9); x0=zeros(2))
        @test_throws DimensionMismatch ApxChol.solve!(zeros(8), S, zeros(9))
        @test_throws ArgumentError ApxChol.solve(S, ones(9); maxiter=-1)
        @test_throws ArgumentError ApxCholSolver(grid_laplacian(3, 3); sampler=:nope)
        @test_throws ArgumentError ApxCholSolver(grid_laplacian(3, 3); backend=:cuda)
        @test_throws ArgumentError ApxCholSolver(grid_laplacian(3, 3); factor_storage=Float64)
        @test_throws InexactError ApxCholSolver(grid_laplacian(3, 3); seed=-1)
        for kw in ((threads=-1,), (rtol=0.0,), (rtol=NaN,), (degree_quantile=2.0,))
            e = caught(() -> ApxCholSolver(grid_laplacian(3, 3); kw...))
            @test e isa ApxCholError && e.status == :invalid_argument
        end
        e = caught(() -> ApxChol.solve(S, fill(NaN, 9)))
        @test e isa ApxCholError && e.status == :invalid_argument && occursin("non-finite", e.message)
        adjacency = sparse([0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0])
        e = caught(() -> ApxCholSolver(adjacency))
        @test e isa ApxCholError && e.status == :invalid_operator
        @test occursin("positive diagonal", e.message)
        @test startswith(sprint(showerror, e), "ApxCholError(:invalid_operator): ")
    end

    @testset "factor storage" begin
        A = grid_laplacian(15, 15; shift=0.01)
        b = ones(size(A, 1))
        S = ApxCholSolver(A; factor_storage=Float32)
        @test ApxChol.stats(S).factor_storage == Float32
        @test ApxChol.options(S).factor_storage == Float32
        half = caught(() -> ApxCholSolver(A; factor_storage=Float16, backend=:cpu))
        if half === nothing
            H = ApxCholSolver(A; factor_storage=Float16, backend=:cpu)
            @test ApxChol.stats(H).factor_storage == Float16
            @test ApxChol.solve(H, b).converged
        else
            @test half isa ApxCholError && half.status == :unsupported      # no FP16 on this CPU
        end
        withenv("APXCHOL_FACTOR_STORAGE" => "half") do      # no longer read by apxchol
            @test ApxChol.stats(ApxCholSolver(A)).factor_storage == Float32
        end
    end

    @testset "lumping of positive off-diagonals" begin
        A = grid_laplacian(4, 4; shift=0.5)
        A[2, 1] = -A[2, 1]; A[1, 2] = -A[1, 2]
        S = ApxCholSolver(A)
        s = ApxChol.stats(S)
        @test s.lumped_offdiag == 2 && s.sddm
        b = ones(size(A, 1))
        r = ApxChol.solve(S, b; rtol=1e-10)
        @test r.converged && relres(A, r.x, b) < 1e-9          # against the original A
    end

    @testset "extreme scales are solved as their scaled problem" begin
        for shift in (0.0, 0.05)
            A = grid_laplacian(12, 12; shift)
            b = compatible_rhs(size(A, 1); salt=2)
            S = ApxCholSolver(A; backend=:cpu, threads=1)
            ref = ApxChol.solve(S, b)
            for e in (-700, 700)
                r = ApxChol.solve(S, ldexp.(b, e))
                @test r.converged && r.iterations == ref.iterations
                @test bits(r.x) == bits(ldexp.(ref.x, e))
                # The same x; apxchol recomputes its residual at the input's scale.
                @test r.relative_residual ≈ ref.relative_residual rtol = 4eps()
            end
        end
    end

    @testset "the reported residual is the original system's" begin
        for shift in (0.0, 0.05)
            A = grid_laplacian(16, 15; shift)
            b = compatible_rhs(size(A, 1); salt=4)
            S = ApxCholSolver(A)
            r = ApxChol.solve(S, b)
            @test abs(r.relative_residual - relres(A, r.x, b)) <= residual_rounding(A, r.x, b)
            @test ApxChol.solve(S, b; maxiter=0).relative_residual == 1.0
            @test ApxChol.solve(S, zeros(size(A, 1))).relative_residual == 0.0
        end
    end

    @testset "preconditioner for Krylov.jl" begin
        A = grid_laplacian(40, 40; shift=1e-3)
        b = compatible_rhs(size(A, 1)) .+ 0.5
        S = ApxCholSolver(A)
        P = ApxChol.aspreconditioner(S)
        @test P isa ApxCholPreconditioner && size(P) == size(A) && eltype(P) == Float64
        z = P \ b
        y = zeros(length(b)); ldiv!(y, P, b)
        @test y == z
        w = copy(b); ldiv!(P, w)
        @test w == z
        x, stats = Krylov.cg(A, b; M=P, ldiv=true, rtol=1e-10, atol=0.0)
        @test stats.solved && relres(A, x, b) < 1e-8
        _, plain = Krylov.cg(A, b; rtol=1e-10, atol=0.0)
        @test stats.niter < plain.niter
    end

    @testset "apply: a fixed map; centred for a Laplacian" begin
        r = compatible_rhs(144; salt=3)
        for shift in (0.1, 0.0)
            S = ApxCholSolver(grid_laplacian(12, 12; shift))
            P = ApxChol.aspreconditioner(S)
            z = [P \ r for _ in 1:12]
            @test all(zi -> bits(zi) == bits(z[1]), z)
            ApxChol.solve(S, r)                            # solves do not change it
            @test bits(P \ r) == bits(z[1])
            shift == 0 && @test abs(sum(z[1])) <= 1e-12 * norm(z[1], 1)
        end
    end

    @testset "backends" begin
        if !ApxChol.backend_available(:metal)
            e = caught(() -> ApxCholSolver(grid_laplacian(3, 3); backend=:metal))
            @test e isa ApxCholError && e.status == :unsupported
        end
        @test ApxChol.stats(ApxCholSolver(grid_laplacian(3, 3); backend=:cpu)).backend == :cpu
    end
end
