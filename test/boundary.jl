@testset "Boundary contracts" begin
    @testset "malformed CSC never reaches native memory unchecked" begin
        # Lengths are checked in Julia; index values and monotonicity by apxchol.
        rejected(e) = e isa ArgumentError || (e isa ApxCholError && e.status == :invalid_argument)
        for T in (Float64, Float32), mutate in (
                A -> pop!(A.colptr), A -> push!(A.colptr, A.colptr[end]),
                A -> (A.colptr[1] = 0), A -> (A.colptr[2] = 0),
                A -> (A.colptr[end] += 1), A -> empty!(A.rowval),
                A -> empty!(A.nzval), A -> (A.rowval[1] = 0),
                A -> (A.rowval[1] = 3), A -> (A.colptr[end] = typemax(Int64)))
            A = SparseMatrixCSC{T,Int64}(sparse([3.0 -1.0; -1.0 2.0]))
            mutate(A)
            @test rejected(caught(() -> ApxCholSolver(A)))
            @test rejected(caught(() -> ApxCholFactor(A)))
        end
        # Unsorted columns are canonicalized and duplicates summed.
        A = sparse([3.0 -1.0; -1.0 2.0])
        resize!(A.rowval, 6); resize!(A.nzval, 6)
        A.colptr .= [1, 4, 7]; A.rowval .= [2, 1, 1, 2, 1, 1]
        A.nzval .= [-1.0, 1.0, 2.0, 2.0, -0.5, -0.5]
        S = ApxCholSolver(A; threads=1)
        @test norm([3.0 -1.0; -1.0 2.0] * ApxChol.solve(S, [1.0, 4.0]).x - [1.0, 4.0]) < 1e-7
        close(S)
    end

    @testset "right-hand sides at any scale; honest residuals" begin
        A = sparse([3.0 -1.0; -1.0 2.0])
        S = ApxCholSolver(A; threads=1)
        b = [1.0, 4.0]
        for scale in (1e-300, 1e-200, 1.0, 1e200, 1e300)
            r = ApxChol.solve(S, scale .* b)
            @test r.converged
            @test r.x ≈ scale .* (A \ b) rtol = 1e-8
            @test ApxChol.solve(S, scale .* b; maxiter=0).relative_residual == 1.0
        end
        # Subnormal entries: the returned x is subnormal too, and its residual says so.
        r = ApxChol.solve(S, 1e-320 .* b)
        @test r.iterations > 0 && norm(r.x) > 0 && r.converged == (r.relative_residual < 1e-8)
        @test ApxChol.solve(S, zeros(2)).relative_residual == 0.0
        x = copy(b)
        r = ApxChol.solve!(x, S, x; x0=x)
        @test r.relative_residual ≈ norm(b - A * x) / norm(b) atol = 1e-15
        B = hcat(1e-200 .* b, b, zeros(2), 1e200 .* b); X = copy(B)
        R = ApxChol.solve!(X, S, X; maxiter=0)
        @test R.relative_residuals == [1.0, 1.0, 0.0, 1.0] && R.converged == [false, false, true, false]
        R = ApxChol.solve(S, B)
        @test all(R.converged)
        @test_throws ApxCholError ApxChol.solve(S, [NaN, 0.0])
        @test_throws ApxCholError ApxChol.apply!(zeros(2), S, [Inf, 0.0])
        # A huge guess on a normal b is solved (apxchol rescales b and x0 together)
        # with a finite certified residual; a pair that cannot share one scale is not.
        r = ApxChol.solve(S, b; x0=fill(0x1p300, 2))
        @test all(isfinite, r.x) && isfinite(r.relative_residual)
        @test r.converged == (r.relative_residual < 1e-8)
        @test r.relative_residual ≈ norm(b - A * r.x) / norm(b) rtol = 1e-12
        e = caught(() -> ApxChol.solve(S, 1e-200 .* b; x0=1e200 .* b))
        @test e isa ApxCholError && e.status == :invalid_argument
        @test occursin("scaling range", e.message)
        close(S)
    end

    @testset "unrepresentable solutions are unsupported" begin
        for d in (1e-30, 1e30)
            A = sparse(d * I, 4, 4)
            S = ApxCholSolver(A; backend=:cpu, threads=1)
            b = fill(d < 1 ? 1e300 : 1e-300, 4)             # x = b / d is not finite or is zero
            e = caught(() -> ApxChol.solve(S, b))
            @test e isa ApxCholError && e.status == :unsupported && !isempty(e.message)
            @test_throws ApxCholError S \ b
            B = hcat(fill(d, 4), b)
            e = caught(() -> ApxChol.solve(S, B))
            @test e isa ApxCholError && e.status == :unsupported
            @test ApxChol.solve(S, B[:, 1]).x ≈ ones(4) rtol = 1e-8
            close(S)
        end
    end
end
