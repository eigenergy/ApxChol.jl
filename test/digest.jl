function fnv1a(v::AbstractVector)
    h = 0xcbf29ce484222325
    for byte in reinterpret(UInt8, v)
        h = (h ⊻ byte) * 0x00000100000001b3
    end
    return h
end
hex(x::Unsigned) = string(x; base=16, pad=16)

# test/digest/digest.cpp against the installed core in deps/usr, with the
# compiler and OpenMP runtime the module was built with.
function build_digest(dir)
    cmake = Sys.which("cmake")
    info = ApxChol.build_info()
    cmake === nothing && return nothing
    omp = Sys.isapple() ? "-Xclang -fopenmp" : "-fopenmp"
    build = joinpath(dir, "build")
    run(pipeline(`$cmake -S $(joinpath(@__DIR__, "digest")) -B $build -DCMAKE_BUILD_TYPE=Release
                  -DCMAKE_CXX_COMPILER=$(info["compiler"])
                  -DCMAKE_PREFIX_PATH=$(dirname(dirname(ApxChol.libapxchol_jl)))
                  -DOpenMP_CXX_FLAGS=$omp -DOpenMP_CXX_LIB_NAMES=omp
                  -DOpenMP_omp_LIBRARY=$(LLVMOpenMP_jll.libomp_path)
                  -DOpenMP_CXX_INCLUDE_DIR=$(joinpath(LLVMOpenMP_jll.artifact_dir, "include"))`;
                 stdout=devnull))
    run(pipeline(`$cmake --build $build`; stdout=devnull))
    return joinpath(build, "digest")
end

@testset "Julia and apxchol's own C++ agree byte for byte" begin
    dir = mktempdir()
    exe = build_digest(dir)
    if exe === nothing
        @warn "No CMake; the C++ comparison did not run"
        @test_broken false
    else
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
