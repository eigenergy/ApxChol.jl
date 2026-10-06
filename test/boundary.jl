@testset "Boundary contracts" begin
    @testset "all ABI fields and enums" begin
        @test [fieldoffset(ApxChol.COptions, i) for i in 1:14] ==
              [0, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64]
        @test [fieldoffset(ApxChol.CSolveInfo, i) for i in 1:5] == [0, 4, 8, 16, 24]
        @test [fieldoffset(ApxChol.CStats, i) for i in 1:10] ==
              [0, 4, 8, 16, 24, 32, 40, 48, 56, 60]
        @test ApxChol.STATUS == (:success, :not_converged, :invalid_argument,
              :invalid_operator, :no_factor_values, :unsupported, :out_of_memory, :internal_error)
        @test ApxChol.BACKENDS == (:cpu, :metal)
        @test ApxChol.SAMPLERS == (:gks, :trace_cycle)
        @test ApxChol.PARTITIONERS == (:block_greedy, :priority_greedy, :baumann_kyng)
        @test ApxChol.STORAGES == Dict(:vec => 0, :bstr => 2, :vec_pool_aos => 4)
    end
    @testset "malformed mutable CSC rejected before native access" begin
        for T in (Float64, Float32), mutate in (
                A -> pop!(A.colptr), A -> push!(A.colptr, A.colptr[end]),
                A -> (A.colptr[1] = 0), A -> (A.colptr[2] = 0),
                A -> (A.colptr[end] += 1), A -> empty!(A.rowval),
                A -> empty!(A.nzval), A -> (A.rowval[1] = 0),
                A -> (A.rowval[1] = 3), A -> (A.colptr[end] = typemax(Int64)))
            A = SparseMatrixCSC{T,Int64}(sparse([3.0 -1.0; -1.0 2.0]))
            mutate(A)
            @test_throws ArgumentError ApxCholSolver(A)
            @test_throws ArgumentError ApxCholFactor(A)
        end
        # Native import still canonicalizes unsorted columns and sums duplicates.
        A = sparse([3.0 -1.0; -1.0 2.0])
        resize!(A.rowval, 6); resize!(A.nzval, 6)
        A.colptr .= [1, 4, 7]; A.rowval .= [2, 1, 1, 2, 1, 1]
        A.nzval .= [-1.0, 1.0, 2.0, 2.0, -0.5, -0.5]
        S = ApxCholSolver(A; threads=1)
        @test norm([3.0 -1.0; -1.0 2.0] * solve(S, [1., 4.]).x - [1., 4.]) < 1e-7
        close(S)
    end
    @testset "original residual and exact aliasing" begin
        A = sparse([3.0 -1.0; -1.0 2.0]); S = ApxCholSolver(A; threads=1)
        for scale in (1e-320, 1e-200, 1.0, 1e200)
            b = scale .* [1.0, 4.0]
            r = solve(S, b; maxiter=0)
            @test !r.converged && r.relative_residual ≈ 1.0
        end
        tiny = solve(S, 1e-200 .* [1.0, 4.0])
        @test !tiny.converged && tiny.relative_residual == 1.0
        @test_throws NotConvergedError S \ (1e-200 .* [1.0, 4.0])
        b = [1., 4.]; x = copy(b)
        r = solve!(x, S, x; x0=x)
        @test r.relative_residual ≈ norm(b-A*x)/norm(b) atol=1e-15
        z = copy(b); expected = ApxChol.apply!(similar(b), S, b)
        ApxChol.apply!(z, S, z)
        @test reinterpret(UInt64, z) == reinterpret(UInt64, expected)
        B = hcat(1e-200 .* b, b, zeros(2)); X = copy(B)
        R = solve!(X, S, X; maxiter=0)
        @test R.relative_residuals ≈ [1., 1., 0.]
        @test R.converged == [false, false, true]
        @test solve(S, zeros(2)).relative_residual == 0.0
        @test_throws ApxCholError solve(S, [NaN, 0.])
        @test_throws ApxCholError ApxChol.apply!(z, S, [Inf, 0.])
        # The reused error text must be consumed under the same lock as the call.
        tasks = map(1:8) do i
            Threads.@spawn begin
                try
                    solve(S, i % 2 == 0 ? [NaN, 0.] : b; rtol=i % 2 == 0 ? nothing : 0.)
                    false
                catch e
                    e isa ApxCholError && occursin(i % 2 == 0 ? "non-finite" : "tol", e.message)
                end
            end
        end
        @test all(fetch.(tasks))
        # Warmed apply! should allocate no Julia error vector per iteration.
        ApxChol.apply!(z, S, b)
        allocation(S, z, b) = @allocated ApxChol.apply!(z, S, b)
        allocation(S, z, b)
        @test allocation(S, z, b) < 256
        close(S); close(S)
        @test_throws ArgumentError ApxChol.apply!(z, S, b)
    end
end

@testset "Metal norm range guard" begin
    if ApxChol.backend_available(:metal)
        S = ApxCholSolver(sparse([3.0 -1.0; -1.0 2.0]); backend=:metal, threads=1)
        for scale in (1e-200, 1e200)
            for b in (scale .* [1., 4.], reshape(scale .* [1., 4.], 2, 1))
                e = try solve(S, b); nothing catch err; err end
                @test e isa ApxCholError && e.status == :unsupported
            end
        end
        close(S)
    else
        @test_skip false
    end
end
