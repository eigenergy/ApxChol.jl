function fnv1a(v::AbstractVector)
    h = 0xcbf29ce484222325
    for byte in reinterpret(UInt8, v)
        h = (h ⊻ byte) * 0x00000100000001b3
    end
    return h
end
hex(x::Unsigned) = string(x; base=16, pad=16)

@testset "Julia and C agree byte for byte on the same library" begin
    cc = Sys.which("cc")
    include_dir = get(ApxChol.build_info(), "include_dir", "")
    if cc === nothing || !isdir(include_dir)
        @warn "No C compiler or apxchol include directory; the C comparison did not run"
        @test_broken false
    else
        dir = mktempdir()
        exe = joinpath(dir, "digest")
        libdir = dirname(ApxChol.libapxchol)
        run(`$cc -std=c11 -I$include_dir $(joinpath(@__DIR__, "digest.c")) -L$libdir -lapxchol
             -Wl,-rpath,$libdir -o $exe`)
        for shift in (0.0, 0.03)
            A = grid_laplacian(23, 17; shift)
            n = size(A, 1)
            b = compatible_rhs(n; salt=2) .+ shift
            input = joinpath(dir, "system.bin")
            open(input, "w") do io
                write(io, Int64[n, nnz(A)], A.colptr, A.rowval, A.nzval, b)
            end
            S = ApxCholSolver(A; backend=:cpu, threads=1, seed=7, keep_factor=true,
                              factor_storage=Float32)
            r = ApxChol.solve(S, b)
            f = ApxChol.factor(S)
            julia = join(["colptr $(hex(fnv1a(f.L.colptr)))", "rowval $(hex(fnv1a(f.L.rowval)))",
                          "nzval $(hex(fnv1a(f.L.nzval)))", "perm $(hex(fnv1a(invperm(f.p))))",
                          "x $(hex(fnv1a(r.x)))", "iterations $(r.iterations)",
                          "residual $(hex(reinterpret(UInt64, r.relative_residual)))"], "\n")
            @test readchomp(`$exe $input`) == julia
            close(S)
        end
    end
end
