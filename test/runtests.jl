using ApproxCholesky
using LinearAlgebra
using SparseArrays
using Test

@testset "Laplacian solve" begin
    A = sparse([
        1.0 -1.0  0.0  0.0
       -1.0  2.0 -1.0  0.0
        0.0 -1.0  2.0 -1.0
        0.0  0.0 -1.0  1.0
    ])
    b = [1.0, -1.0, -1.0, 1.0]
    for threads in (1, min(4, Sys.CPU_THREADS))
        result = solve(A, b; tolerance=1e-10, threads=threads, seed=7)
        @test norm(A * result.x - b) / norm(b) < 1e-10
        @test result.converged
        @test result.iterations >= 1
        @test result.setup_seconds >= 0
        @test result.solve_seconds >= 0
    end
end

@testset "SDDM solve" begin
    A = sparse([3.0 -1.0; -1.0 2.0])
    b = [1.0, 4.0]
    result = solve(A, b; tolerance=1e-11, threads=2)
    @test result.x ≈ A \ b rtol=1e-10
end

@testset "parallel runtime smoke test" begin
    n = 4096
    A = spdiagm(-1 => -ones(n - 1), 0 => [1.0; fill(2.0, n - 2); 1.0],
                1 => -ones(n - 1))
    b = [isodd(i) ? 1.0 : -1.0 for i in 1:n]
    result = solve(A, b; tolerance=1e-9, threads=min(4, Sys.CPU_THREADS))
    @test norm(A * result.x - b) / norm(b) < 1e-9
end

@testset "argument checks" begin
    @test_throws DimensionMismatch solve(spzeros(2, 3), zeros(2))
    @test_throws DimensionMismatch solve(spdiagm(0 => ones(2)), zeros(3))
    @test_throws ArgumentError solve(spdiagm(0 => ones(2)), zeros(2); threads=-1)
end
