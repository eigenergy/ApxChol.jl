@testset "Memory and thread safety" begin
    @testset "warmed solves and applications allocate nothing" begin
        A = grid_laplacian(20, 20; shift=0.01)
        n = size(A, 1)
        S = ApxCholSolver(A; threads=1)
        P = ApxChol.aspreconditioner(S)
        b = compatible_rhs(n) .+ 0.1
        x = zeros(n); z = zeros(n)
        B = hcat(b, b); X = zeros(n, 2)
        solve_vector(x, S, b) = @allocated ApxChol.solve!(x, S, b)
        solve_column(X, S, B) = @allocated ApxChol.solve!(view(X, :, 2), S, view(B, :, 1))
        divide(x, S, b) = @allocated ldiv!(x, S, b)
        apply(z, S, b) = @allocated ApxChol.apply!(z, S, b)
        precondition(z, P, b) = @allocated ldiv!(z, P, b)
        for (f, args) in ((solve_vector, (x, S, b)), (solve_column, (X, S, B)), (divide, (x, S, b)),
                          (apply, (z, S, b)), (precondition, (z, P, b)))
            f(args...)
            @test f(args...) == 0
        end
    end

    @testset "close waits for in-flight calls; later calls throw" begin
        A = grid_laplacian(30, 30; shift=0.01)
        b = ones(size(A, 1))
        for _ in 1:20
            S = ApxCholSolver(A)
            outcomes = fetch.(map(1:6) do t
                Threads.@spawn begin
                    x = zeros(size(A, 1))
                    seen = Symbol[]
                    for _ in 1:20
                        e = caught(() -> ApxChol.solve!(x, S, b))
                        push!(seen, e === nothing ? :solved : e isa InvalidStateException ? :closed : :other)
                        t == 1 && length(seen) == 3 && close(S)
                    end
                    seen
                end
            end)
            @test all(o -> o in (:solved, :closed), reduce(vcat, outcomes))
            @test !isopen(S)
        end
    end

    @testset "factor reads overlap; close waits for them" begin
        A = grid_laplacian(25, 25; shift=0.05)
        b = ones(size(A, 1))
        F = ApxCholFactor(A; threads=1)
        reference = ApxChol.solve(ApxCholSolver(A, F; threads=1), b).x
        solved = fetch.(map(1:4) do _
            Threads.@spawn bits(ApxChol.solve(ApxCholSolver(A, F; threads=1), b).x)
        end)
        @test all(==(bits(reference)), solved)
        for _ in 1:20
            G = ApxCholFactor(A; threads=1)
            outcomes = fetch.(map(1:6) do t
                Threads.@spawn begin
                    t == 1 && (sleep(0.001); close(G); return :closer)
                    e = caught(() -> ApxChol.solve(ApxCholSolver(A, G; threads=1), b))
                    e === nothing ? :adopted : e isa InvalidStateException ? :closed : e
                end
            end)
            @test all(o -> o in (:closer, :adopted, :closed), outcomes)
            @test !isopen(G)
        end
        S = ApxCholSolver(A, F)
        close(F)                                        # a solver holds its own copy
        @test ApxChol.solve(S, b).converged
    end

    @testset "finalizers and close" begin
        A = grid_laplacian(6, 6; shift=0.1)
        for _ in 1:50
            ApxCholSolver(A)
            ApxCholSolver(A, ApxCholFactor(A))
        end
        GC.gc(); GC.gc()
        S = ApxCholSolver(A)
        @test isopen(S)
        close(S); close(S)
        @test !isopen(S) && sprint(show, S) == "ApxCholSolver(36×36, cpu, closed)"
        for use in (() -> ApxChol.solve(S, ones(36)), () -> ApxChol.stats(S),
                    () -> ApxChol.apply!(zeros(36), S, ones(36)), () -> ApxCholFactor(S))
            e = caught(use)
            @test e isa InvalidStateException && e.state == :closed
        end
    end

    @testset "error messages are read under the solver's lock" begin
        S = ApxCholSolver(sparse([3.0 -1.0; -1.0 2.0]); threads=1)
        tasks = map(1:8) do i
            Threads.@spawn begin
                e = caught(() -> iseven(i) ? ApxChol.solve(S, [NaN, 0.0]) :
                                             ApxChol.solve(S, [1.0, 4.0]; rtol=0.0))
                e isa ApxCholError && occursin(iseven(i) ? "non-finite" : "tol", e.message)
            end
        end
        @test all(fetch.(tasks))
    end

    @testset "partially overlapping arrays are copied, exact aliases are not" begin
        A = grid_laplacian(10, 10; shift=0.02)
        n = size(A, 1)
        S = ApxCholSolver(A; threads=1)
        b = compatible_rhs(n) .+ 0.2
        reference = ApxChol.solve(S, b).x
        v = vcat(b, 0.0)
        ApxChol.solve!(view(v, 2:n+1), S, view(v, 1:n))
        @test bits(v[2:end]) == bits(reference)
        z = copy(b)
        @test bits(ApxChol.apply!(z, S, z)) == bits(ApxChol.apply!(zeros(n), S, b))
    end

    @testset "concurrent solvers on Julia threads" begin
        A = grid_laplacian(16, 16)
        results = Vector{Any}(undef, 4)
        Threads.@threads for t in 1:4
            S = ApxCholSolver(A; threads=1)
            results[t] = ApxChol.solve(S, compatible_rhs(size(A, 1); salt=t))
            close(S)
        end
        @test all(r -> r.converged, results)
    end

    @testset "one OpenMP runtime; thread limits are call-scoped" begin
        @test haskey(ENV, "KMP_GTID_MODE")
        before = omp_max_threads()
        omp_set_threads(3)
        @test ApxChol.max_threads() == 3
        S = ApxCholSolver(grid_laplacian(70, 70); threads=2)
        @test ApxChol.stats(S).setup_max_threads == 2
        @test omp_max_threads() == 3
        @test ApxChol.solve(S, compatible_rhs(4900)).converged
        @test omp_max_threads() == 3
        omp_set_threads(before)
        # Julia may load its private libgomp (Linux, Julia 1.13); every other
        # OpenMP runtime in the process must be the JLL's libomp.
        julia_private = realpath(joinpath(Sys.BINDIR, Base.PRIVATE_LIBDIR))
        runtimes = unique(realpath.(filter(Libdl.dllist()) do path
            name = lowercase(basename(path))
            any(p -> startswith(name, p), ("libomp", "libiomp", "libgomp")) &&
                !startswith(realpath(path), julia_private)
        end))
        @test runtimes == [realpath(LLVMOpenMP_jll.libomp_path)]
    end

    @testset "garbage collection on other threads never waits for a native call" begin
        script = joinpath(@__DIR__, "stall.jl")
        out = readchomp(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project())
                         -t 2,1 $script`)
        seconds, stall, safepoint = parse.(Float64, split(out))
        # Without GC safety both are about the longest native call; the stall
        # also includes the collection itself, which a loaded machine stretches.
        @test seconds > 0.5                             # long enough to measure against
        @test safepoint < seconds / 10
        @test stall < seconds / 2
    end
end
