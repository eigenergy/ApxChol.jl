"""
    ApxChol

Julia bindings for [apxchol](https://github.com/AlgOptGroup/apxchol), a parallel
approximate-Cholesky preconditioned conjugate-gradient solver for graph
Laplacian, SDDM and lumpable M-matrix systems, through apxchol's C ABI.

Factor once with [`ApxCholSolver`](@ref), then solve any number of right-hand
sides with `S \\ b`, `ldiv!` or [`ApxChol.solve!`](@ref);
[`ApxChol.aspreconditioner`](@ref) exposes one preconditioner application for
Krylov solvers. An [`ApxCholFactor`](@ref) carries one factorization to any
number of solvers, including solvers for another operator of the same size.
"""
module ApxChol

using LLVMOpenMP_jll: LLVMOpenMP_jll
using Libdl: Libdl
using LinearAlgebra
using SparseArrays

export ApxCholSolver, ApxCholFactor, ApxCholError, NotConvergedError

include("capi.jl")
include("handles.jl")
include("solve.jl")

@static if VERSION >= v"1.11"
    eval(Expr(:public, :solve, :solve!, :apply!, :aspreconditioner, :factor, :ordering, :stats,
              :options, :version, :build_info, :backend_available, :max_threads,
              :openmp_enabled, :SolveResult, :BlockSolveResult, :ExportedFactor,
              :ApxCholPreconditioner))
end

"""
    ApxChol.version() -> String

The apxchol version and git revision the library was built from.
"""
version() = unsafe_string(@ccall libapxchol.apxchol_version()::Cstring)

"""
    ApxChol.openmp_enabled() -> Bool
"""
openmp_enabled() = (@ccall libapxchol.apxchol_openmp_enabled()::Int32) == 1

"""
    ApxChol.max_threads() -> Int

The calling thread's OpenMP team limit (1 in a serial build); solvers created
with `threads=0` use it.
"""
max_threads() = Int(@ccall libapxchol.apxchol_get_max_threads()::Int32)

"""
    ApxChol.backend_available(backend::Symbol) -> Bool

Whether this build and machine support `backend`: `:cpu`, `:auto`, or `:metal`
(Apple silicon builds with a usable GPU).
"""
backend_available(backend::Symbol) =
    haskey(BACKENDS, backend) &&
    (@ccall libapxchol.apxchol_backend_available(code(BACKENDS, backend, "backend")::Int32)::Int32) == 1

"""
    ApxChol.build_info() -> Dict{String,String}

How the library was built: apxchol's source and revision, the compiler, the
device and the OpenMP runtime.
"""
function build_info()
    info = Dict{String,String}()
    isfile(build_info_path) || return info
    for line in eachline(build_info_path)
        key, value = split(line, '='; limit=2)
        info[key] = value
    end
    return info
end

pin_revision() =
    only(m.captures[1] for m in eachmatch(r"^revision=(\w+)"m,
                                          read(joinpath(@__DIR__, "..", "deps", "apxchol.pin"), String)))

function __init__()
    rebuild = "run `import Pkg; Pkg.build(\"ApxChol\")`"
    isfile(libapxchol) || error("ApxChol is not built; $rebuild")
    info = build_info()
    built_omp = get(info, "libomp", "")
    if !isempty(built_omp) && realpath(built_omp) != realpath(LLVMOpenMP_jll.libomp_path)
        error("ApxChol was built against $built_omp, but LLVMOpenMP_jll now provides " *
              "$(LLVMOpenMP_jll.libomp_path); $rebuild")
    end
    if get(info, "apxchol_source", "") == "pin" && get(info, "apxchol_revision", "") != pin_revision()
        error("ApxChol was built from apxchol $(info["apxchol_revision"]), but deps/apxchol.pin " *
              "names $(pin_revision()); $rebuild")
    end
    # While few threads exist, libomp identifies the calling thread by its stack
    # address. Julia tasks run on their own stacks, so concurrent solves on Julia
    # threads could share one OpenMP thread identity and crash. Keyed
    # thread-local lookup (mode 2) does not depend on the stack; it must be
    # chosen before libomp's first OpenMP call. A caller's setting wins.
    haskey(ENV, "KMP_GTID_MODE") || (ENV["KMP_GTID_MODE"] = "2")
    # The library's OpenMP runtime goes in first: two OpenMP runtimes in one
    # process can crash once a parallel region starts.
    Libdl.dlopen(LLVMOpenMP_jll.libomp_path, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    Libdl.dlopen(libapxchol, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    # The struct mirrors in capi.jl must be the library's: its ABI version and
    # options size, and the defaults only this layout produces.
    abi = @ccall libapxchol.apxchol_c_abi_version()::Int32
    defaults = Ref{COptions}()
    status = @ccall libapxchol.apxchol_options_default(defaults::Ref{COptions},
                                                       sizeof(COptions)::Csize_t)::Int32
    if abi != C_ABI_VERSION || status != 0 || defaults[].backend != BACKENDS.auto ||
       defaults[].factor_storage != 0 || defaults[].keep_factor_values ∉ (0, 1)
        error("$libapxchol (apxchol $(version()), C ABI $abi) does not match ApxChol's " *
              "C ABI mirror; $rebuild")
    end
    DEFAULTS[] = defaults[]
    return nothing
end

end
