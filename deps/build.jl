using LLVMOpenMP_jll

const APXCHOL_REVISION = "f1830f24fce12852536f4b6026c0dae99c0015d5"
const APXCHOL_REPOSITORY = "https://github.com/samtalki/apxchol.git"

function checked_source_directory()
    configured = get(ENV, "APXCHOL_SOURCE_DIR", "")
    source = if !isempty(configured)
        abspath(configured)
    else
        checkout = joinpath(@__DIR__, "build", "apxchol-source")
        if !isdir(joinpath(checkout, ".git"))
            mkpath(dirname(checkout))
            run(`git clone --filter=blob:none --no-checkout $APXCHOL_REPOSITORY $checkout`)
        end
        run(`git -C $checkout fetch --depth=1 origin $APXCHOL_REVISION`)
        run(`git -C $checkout checkout --detach FETCH_HEAD`)
        checkout
    end

    isfile(joinpath(source, "include", "apxchol", "c_api.h")) ||
        error("apxchol source at $source does not contain include/apxchol/c_api.h")
    return source
end

cmake = Sys.which("cmake")
cmake === nothing && error("cmake is required to build ApproxCholesky.jl")

source = checked_source_directory()
build = joinpath(@__DIR__, "build", "wrapper")
prefix = joinpath(@__DIR__, "usr")
mkpath(build)
mkpath(prefix)

omp_include = joinpath(LLVMOpenMP_jll.artifact_dir, "include")
omp_library = LLVMOpenMP_jll.libomp_path
config = `$cmake -S $(@__DIR__) -B $build -DCMAKE_BUILD_TYPE=Release -DAPXCHOL_SOURCE_DIR=$source -DAPXCHOL_OPENMP_INCLUDE_DIR=$omp_include -DAPXCHOL_OPENMP_LIBRARY=$omp_library -DCMAKE_INSTALL_PREFIX=$prefix`
run(config)
run(`$cmake --build $build --config Release --parallel`)
run(`$cmake --install $build --config Release`)

library_name = Sys.iswindows() ? "approxcholesky.dll" :
               Sys.isapple() ? "libapproxcholesky.dylib" : "libapproxcholesky.so"
library_path = Sys.iswindows() ? joinpath(prefix, "bin", library_name) :
                                joinpath(prefix, "lib", library_name)
isfile(library_path) || error("wrapper library was not installed at $library_path")
