# Builds deps/usr/lib/libapxchol from the apxchol revision in deps/apxchol.pin
# (or ENV["APXCHOL_SOURCE_DIR"]) against LLVMOpenMP_jll's libomp.
using LLVMOpenMP_jll

const DEPS = @__DIR__

Sys.iswindows() && error("ApxChol.jl supports Linux and macOS only")

function read_pin(path)
    pin = Dict{String,String}()
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, '#')) && continue
        key, value = split(line, '='; limit=2)
        pin[strip(key)] = strip(value)
    end
    return pin
end

function git_revision(dir)
    try
        return readchomp(`git -C $dir rev-parse HEAD`)
    catch
        return "unknown"
    end
end

function source_directory(pin)
    configured = get(ENV, "APXCHOL_SOURCE_DIR", "")
    isempty(configured) || return abspath(configured)
    checkout = joinpath(DEPS, "build", "apxchol-source")
    if !isdir(joinpath(checkout, ".git"))
        rm(checkout; force=true, recursive=true)
        mkpath(dirname(checkout))
        run(`git init -q $checkout`)
        run(`git -C $checkout remote add origin $(pin["repository"])`)
    else
        run(`git -C $checkout remote set-url origin $(pin["repository"])`)
    end
    run(`git -C $checkout fetch -q --depth=1 origin $(pin["revision"])`)
    run(`git -C $checkout checkout -q --detach FETCH_HEAD`)
    head = git_revision(checkout)
    head == pin["revision"] || error("apxchol checkout is at $head, expected $(pin["revision"])")
    return checkout
end

pin = read_pin(joinpath(DEPS, "apxchol.pin"))
source = source_directory(pin)
for required in ("src/factorization.cpp", "src/operator_class.cpp", "src/solve.cpp",
                 "src/c_api.cpp", "include/apxchol/c_api.h",
                 "include/apxchol/version.h.in", "cmake/apxchol_native_arch.cmake")
    isfile(joinpath(source, required)) || error("apxchol source at $source lacks $required")
end
revision = git_revision(source)

cmake = Sys.which("cmake")
cmake === nothing && error("CMake (>= 3.20) is required to build ApxChol.jl")

build = joinpath(DEPS, "build", "libapxchol")
prefix = joinpath(DEPS, "usr")
rm(prefix; force=true, recursive=true)
mkpath(build)

omp_include = joinpath(LLVMOpenMP_jll.artifact_dir, "include")
omp_library = LLVMOpenMP_jll.libomp_path
short_revision = revision == "unknown" ? revision : revision[1:7]
run(`$cmake -S $DEPS -B $build -DCMAKE_BUILD_TYPE=Release
     -DAPXCHOL_SOURCE_DIR=$source -DAPXCHOL_GIT_SHA=$short_revision
     -DAPXCHOL_JL_OPENMP_INCLUDE_DIR=$omp_include
     -DAPXCHOL_JL_OPENMP_LIBRARY=$omp_library
     -DCMAKE_INSTALL_PREFIX=$prefix`)
run(`$cmake --build $build --config Release --parallel`)
run(`$cmake --install $build --config Release`)

library = joinpath(prefix, "lib", Sys.isapple() ? "libapxchol.dylib" : "libapxchol.so")
isfile(library) || error("libapxchol was not installed at $library")
cache = readlines(joinpath(build, "CMakeCache.txt"))
compiler_line = findfirst(l -> startswith(l, "CMAKE_CXX_COMPILER:"), cache)
compiler = compiler_line === nothing ? "unknown" : split(cache[compiler_line], '='; limit=2)[2]
metal = any(l -> startswith(l, "CMAKE_OBJCXX_COMPILER:"), cache) ? "on" : "off"
open(joinpath(prefix, "build_info.txt"), "w") do io
    println(io, "apxchol_source=", isempty(get(ENV, "APXCHOL_SOURCE_DIR", "")) ? pin["repository"] : source)
    println(io, "apxchol_revision=", revision)
    println(io, "libomp=", omp_library)
    println(io, "compiler=", compiler)
    println(io, "metal=", metal)
end
