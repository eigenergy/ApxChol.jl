# The C ABI (apxchol/c_api.h, ABI version 1) mirrored one to one: structs,
# enums, statuses, and the macro every native call goes through.

const libapxchol = normpath(joinpath(@__DIR__, "..", "deps", "usr", "lib",
                                     Sys.isapple() ? "libapxchol.dylib" : "libapxchol.so"))
const build_info_path = normpath(joinpath(@__DIR__, "..", "deps", "usr", "build_info.txt"))
const C_ABI_VERSION = 1

struct COptions                        # 80 bytes (4 bytes of padding after keep_factor_values)
    struct_size::UInt32
    backend::Int32
    tol::Float64
    max_iter::Int32
    stagnation_window::Int32
    seed::UInt32
    threads::Int32
    sampler::Int32
    partitioner::Int32
    keep_factor_values::Int32
    degree_quantile::Float64
    exact_clique_max_degree::UInt64
    omp_threshold::UInt64
    factor_storage::Int32
end

struct CSolveInfo                      # 32 bytes
    struct_size::UInt32
    converged::Int32
    iterations::Int64
    relative_residual::Float64
    solve_seconds::Float64
end
CSolveInfo() = CSolveInfo(sizeof(CSolveInfo), 0, 0, 0.0, 0.0)

struct CStats                          # 72 bytes
    struct_size::UInt32
    backend::Int32
    n::Int64
    factor_nnz::Int64
    lumped_offdiag::Int64
    rounds::Int64
    peak_graph_bytes::Int64
    setup_seconds::Float64
    sddm::Int32
    setup_max_threads::Int32
    factor_storage::Int32
end
CStats() = CStats(sizeof(CStats), 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0)

const STATUS = (:success, :not_converged, :invalid_argument, :invalid_operator,
                :no_factor_values, :unsupported, :out_of_memory, :internal_error)
const BACKENDS = (cpu=0, metal=1, auto=2)
const SAMPLERS = (gks=0, trace_cycle=1)
const PARTITIONERS = (block_greedy=0, priority_greedy=1, baumann_kyng=2)

status_symbol(c::Integer) = 0 <= c < length(STATUS) ? STATUS[c + 1] : :unknown

function code(table::NamedTuple, value::Symbol, what)
    haskey(table, value) ||
        throw(ArgumentError("unknown $what :$value; valid: $(join(keys(table), ", "))"))
    return Int32(table[value])
end
symbol(table::NamedTuple, c::Integer) = keys(table)[findfirst(==(c), values(table))]

# Triangular-solve factor storage: `nothing` (AUTO), Float16 or Float32.
storage_code(::Nothing) = Int32(0)
storage_code(::Type{Float16}) = Int32(16)
storage_code(::Type{Float32}) = Int32(32)
storage_code(x) = throw(ArgumentError("factor_storage must be nothing, Float16 or Float32, got $x"))
storage_type(c::Integer) = c == 16 ? Float16 : c == 32 ? Float32 : nothing

"""
    @native libapxchol.f(args...)::T

`@ccall` that, on Julia 1.12 and later, is GC-safe: other Julia threads can
collect garbage while a long factorization or solve runs. apxchol never calls
back into Julia and every array it reads is rooted by the caller, so this is
sound.
"""
macro native(call)
    VERSION >= v"1.12" ? esc(:(@ccall gc_safe = true $call)) : esc(:(@ccall $call))
end

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

const ERROR_CAPACITY = 2048

error_text(buffer::Vector{UInt8}) =
    String(buffer[1:something(findfirst(iszero, buffer), length(buffer) + 1) - 1])

# SUCCESS returns; so does NOT_CONVERGED where the caller reports it in a result.
function check(status::Int32, buffer::Vector{UInt8}; not_converged_ok::Bool=false)
    (status == 0 || (not_converged_ok && status == 1)) && return nothing
    throw(ApxCholError(status_symbol(status), error_text(buffer)))
end
