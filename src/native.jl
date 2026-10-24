# The CxxWrap module deps/apxchol_jl.cpp builds, checked against this Julia and
# the runtimes it was built with; the option codes it takes and the statuses
# its errors carry.

const libapxchol_jl = normpath(joinpath(@__DIR__, "..", "deps", "usr", "lib",
                                        Sys.isapple() ? "libapxchol_jl.dylib" : "libapxchol_jl.so"))
const build_info_path = normpath(joinpath(@__DIR__, "..", "deps", "usr", "build_info.txt"))

"""
    ApxChol.build_info() -> Dict{String,String}

How the library was built: apxchol's source and revision, the device, the
compiler, the Julia minor version and the CxxWrap and OpenMP runtimes.
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

same_file(a, b) = ispath(a) && ispath(b) && realpath(a) == realpath(b)

# The module links libjulia and libcxxwrap_julia, so it loads only into the
# Julia minor version and CxxWrap runtime it was built for; checked before
# CxxWrap opens it, when precompiling and at load.
function check_build()
    rebuild = "run `import Pkg; Pkg.build(\"ApxChol\")`"
    isfile(libapxchol_jl) || error("ApxChol is not built; $rebuild")
    info = build_info()
    built_julia, julia = get(info, "julia", "unknown"), "$(VERSION.major).$(VERSION.minor)"
    built_julia == julia ||
        error("ApxChol was built for Julia $built_julia, not this Julia $julia; each Julia " *
              "minor version needs its own build: $rebuild")
    for (key, provided) in ("libcxxwrap_julia" => CxxWrap.prefix_path(),
                            "libomp" => LLVMOpenMP_jll.libomp_path)
        built = get(info, key, "unknown")
        same_file(built, provided) ||
            error("ApxChol was built against $key $built, but this environment provides " *
                  "$provided; $rebuild")
    end
    if get(info, "apxchol_source", "") == "pin" && get(info, "apxchol_revision", "") != pin_revision()
        error("ApxChol was built from apxchol $(info["apxchol_revision"]), but deps/apxchol.pin " *
              "names $(pin_revision()); $rebuild")
    end
    return nothing
end

function load_openmp()
    # While few threads exist, libomp identifies the calling thread by its stack
    # address. Julia tasks run on their own stacks, so concurrent solves on Julia
    # threads could share one OpenMP thread identity and crash. Keyed
    # thread-local lookup (mode 2) does not depend on the stack; it must be
    # chosen before libomp's first OpenMP call. A caller's setting wins.
    haskey(ENV, "KMP_GTID_MODE") || (ENV["KMP_GTID_MODE"] = "2")
    # The library's OpenMP runtime goes in first: two OpenMP runtimes in one
    # process can crash once a parallel region starts.
    Libdl.dlopen(LLVMOpenMP_jll.libomp_path, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    return nothing
end

"""
The functions and types of deps/apxchol_jl.cpp. Every call takes raw pointers
to arrays its caller has validated and roots; long calls let other Julia
threads collect garbage meanwhile.
"""
module Native

using CxxWrap
import ..ApxChol

ApxChol.check_build()
@wrapmodule(() -> ApxChol.libapxchol_jl)

function __init__()
    ApxChol.check_build()
    ApxChol.load_openmp()
    @initcxx
end

end

const SolverPtr = CxxPtr{Native.Solver}
const FactorPtr = CxxPtr{Native.Factor}
const NO_FACTOR = CxxWrap.ConstCxxPtr{Native.Factor}(C_NULL)  # factorize instead
isnull(p::Union{SolverPtr,FactorPtr}) = p.cpp_object == C_NULL

# The codes deps/apxchol_jl.cpp takes.
const BACKENDS = (cpu=0, metal=1, auto=2)
const SAMPLERS = (gks=0, trace_cycle=1)
const PARTITIONERS = (block_greedy=0, priority_greedy=1, baumann_kyng=2)

function code(table::NamedTuple, value::Symbol, what)
    haskey(table, value) ||
        throw(ArgumentError("unknown $what :$value; valid: $(join(keys(table), ", "))"))
    return Int32(table[value])
end
symbol(table::NamedTuple, c::Integer) = keys(table)[findfirst(==(c), values(table))]

# Triangular-solve factor storage: `nothing` (the backend's default), Float16 or Float32.
storage_code(::Nothing) = Int32(0)
storage_code(::Type{Float16}) = Int32(16)
storage_code(::Type{Float32}) = Int32(32)
storage_code(x) = throw(ArgumentError("factor_storage must be nothing, Float16 or Float32, got $x"))
storage_type(c::Integer) = c == 16 ? Float16 : c == 32 ? Float32 : nothing

"""
    ApxCholError(status, message)

An error status from apxchol: `:invalid_argument`, `:invalid_operator` (the
matrix violates the Laplacian/SDDM/M-matrix contract), `:no_factor_values`,
`:unsupported` (a backend this build or machine lacks, an input or option the
backend cannot take, or a solution outside the finite range), `:out_of_memory`
or `:internal_error`.
"""
struct ApxCholError <: Exception
    status::Symbol
    message::String
end
Base.showerror(io::IO, e::ApxCholError) = print(io, "ApxCholError(:", e.status, "): ", e.message)

const STATUS = (:invalid_argument, :invalid_operator, :no_factor_values, :unsupported,
                :out_of_memory, :internal_error)

# deps/apxchol_jl.cpp rethrows every C++ exception as "<status>: <message>",
# which CxxWrap raises as an ErrorException; anything else is CxxWrap's own.
function native_error(e::ErrorException)
    m = match(r"^(\w+): ", e.msg)
    (m === nothing || Symbol(m.captures[1]) ∉ STATUS) && return e
    return ApxCholError(Symbol(m.captures[1]), e.msg[ncodeunits(m.match)+1:end])
end

"""
    @native Native.f(args...)

A call into deps/apxchol_jl.cpp, with its errors as [`ApxCholError`](@ref).
"""
macro native(call)
    return quote
        try
            $(esc(call))
        catch e
            e isa ErrorException ? throw(native_error(e)) : rethrow()
        end
    end
end
