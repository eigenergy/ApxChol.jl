# Builds deps/usr/lib/libapxchol_jl, the CxxWrap module, for the running Julia:
# installs the C++ core of the apxchol revision in deps/apxchol.pin (or
# ENV["APXCHOL_SOURCE_DIR"]) into deps/usr, then builds apxchol_jl.cpp against
# it, both against LLVMOpenMP_jll's libomp.
using CxxWrap
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

git_revision(dir) = try readchomp(`git -C $dir rev-parse HEAD`) catch; "unknown" end

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
from_pin = isempty(get(ENV, "APXCHOL_SOURCE_DIR", ""))
source = source_directory(pin)
for required in ("cmake/core/CMakeLists.txt", "cmake/apxcholConfig.cmake.in",
                 "include/apxchol/solver/solve.h")
    isfile(joinpath(source, required)) || error("apxchol source at $source lacks $required")
end
revision = git_revision(source)

cmake = Sys.which("cmake")
cmake === nothing && error("CMake (>= 3.20) is required to build ApxChol.jl")
ninja = Sys.which("ninja") === nothing ? `` : `-G Ninja`

build = joinpath(DEPS, "build")
prefix = joinpath(DEPS, "usr")
for dir in ("core", "eigen", "glue")
    rm(joinpath(build, dir); force=true, recursive=true)
end
rm(prefix; force=true, recursive=true)

# Apple silicon builds the Metal backend; elsewhere the CPU route only.
device = Sys.isapple() && Sys.ARCH === :aarch64 ? "METAL" : "CPU"
# LLVMOpenMP_jll is the one OpenMP runtime: its header and library, with the
# compile flag only, so no compiler links its own runtime (never libgomp). The
# installed core and its consumer must agree on it (apxcholConfig.cmake checks).
openmp = `-DOpenMP_CXX_FLAGS=$(Sys.isapple() ? "-Xclang -fopenmp" : "-fopenmp")
          -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=$(LLVMOpenMP_jll.libomp_path)
          -DOpenMP_CXX_INCLUDE_DIR=$(joinpath(LLVMOpenMP_jll.artifact_dir, "include"))`

# apxchol's installable core, with the Eigen it fetches (never a system one).
core = joinpath(build, "core")
run(`$cmake -S $(joinpath(source, "cmake", "core")) -B $core $ninja -DCMAKE_BUILD_TYPE=Release
     -DAPXCHOL_DEVICE=$device -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=ON $openmp
     -DCMAKE_INSTALL_PREFIX=$prefix`)
run(`$cmake --build $core --config Release --parallel`)
run(`$cmake --install $core --config Release`)

# The installed apxcholConfig.cmake calls find_dependency(Eigen3 CONFIG) even
# when the core used the Eigen it fetched, which installs no Eigen3Config. Until
# apxchol installs or locates that Eigen itself, install the fetched copy (headers
# and its CMake package only) into the same prefix.
eigen_source = joinpath(core, "_deps", "eigen-src")
isfile(joinpath(eigen_source, "CMakeLists.txt")) || error("apxchol's fetched Eigen is not at $eigen_source")
eigen = joinpath(build, "eigen")
run(`$cmake -S $eigen_source -B $eigen $ninja -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF
     -DEIGEN_BUILD_PKGCONFIG=OFF -DCMAKE_INSTALL_PREFIX=$prefix`)
run(`$cmake --install $eigen`)

# The CxxWrap module links libjulia: it is built for this Julia minor only.
# FindJulia must see this Julia, not the first one on PATH.
glue = joinpath(build, "glue")
run(`$cmake -S $DEPS -B $glue $ninja -DCMAKE_BUILD_TYPE=Release $openmp
     -DCMAKE_PREFIX_PATH=$(join([prefix, CxxWrap.prefix_path()], ';'))
     -DJulia_EXECUTABLE=$(joinpath(Sys.BINDIR, "julia")) -DCMAKE_INSTALL_PREFIX=$prefix`)
run(`$cmake --build $glue --config Release --parallel`)
run(`$cmake --install $glue --config Release`)

library = joinpath(prefix, "lib", Sys.isapple() ? "libapxchol_jl.dylib" : "libapxchol_jl.so")
isfile(library) || error("libapxchol_jl was not installed at $library")
cache = readlines(joinpath(glue, "CMakeCache.txt"))
compiler_line = findfirst(l -> startswith(l, "CMAKE_CXX_COMPILER:"), cache)
compiler = compiler_line === nothing ? "unknown" : split(cache[compiler_line], '='; limit=2)[2]
open(joinpath(prefix, "build_info.txt"), "w") do io
    println(io, "apxchol_source=", from_pin ? "pin" : source)
    println(io, "apxchol_repository=", from_pin ? pin["repository"] : source)
    println(io, "apxchol_revision=", revision)
    println(io, "device=", device)
    println(io, "compiler=", compiler)
    println(io, "julia=", VERSION.major, ".", VERSION.minor)
    println(io, "libcxxwrap_julia=", CxxWrap.prefix_path())
    println(io, "libomp=", LLVMOpenMP_jll.libomp_path)
end
