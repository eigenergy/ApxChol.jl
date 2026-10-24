// CxxWrap module over apxchol::solver and apxchol::factorization, the calls
// ApxChol.jl makes (src/native.jl). Julia validates array lengths and passes
// raw pointers to arrays it roots; this file validates everything else the C++
// core does not, classifies every exception into an ApxCholError status, and
// lets other Julia threads collect garbage while a long call runs.
#include <jlcxx/jlcxx.hpp>
#include <jlcxx/tuple.hpp>

#include <apxchol/detail/csc_import.h>
#include <apxchol/operator_class.h>
#include <apxchol/solver/detail/solve_backend.h>
#include <apxchol/solver/detail/thread_budget.h>
#include <apxchol/solver/factorization.h>
#include <apxchol/solver/solve.h>
#include <apxchol/solver/sptrsv/omp.h>
#include <apxchol/version.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>

#include <omp.h>

namespace {

using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

// ApxChol.STATUS: the symbols ApxCholError carries.
enum class status { invalid_argument, invalid_operator, no_factor_values, unsupported,
                    out_of_memory, internal_error };

const char* status_name(status s) {
    switch (s) {
    case status::invalid_argument: return "invalid_argument";
    case status::invalid_operator: return "invalid_operator";
    case status::no_factor_values: return "no_factor_values";
    case status::unsupported: return "unsupported";
    case status::out_of_memory: return "out_of_memory";
    case status::internal_error: return "internal_error";
    }
    return "internal_error";
}

/// A failure with its own status (never re-classified).
struct status_error : std::runtime_error {
    status code;
    status_error(status s, const std::string& message) : std::runtime_error(message), code(s) {}
};

[[noreturn]] void fail(status s, const std::string& message) { throw status_error(s, message); }

void require(bool condition, const char* message) {
    if (!condition) fail(status::invalid_argument, message);
}

/// Runs `body` and rethrows every exception as "<status>: <message>", which
/// CxxWrap turns into an ErrorException and src/native.jl into an
/// ApxCholError. During create, a std::invalid_argument comes from the
/// operator contract (the options were validated before); elsewhere it is an
/// argument error.
template <class Body>
auto guarded(bool creating, Body&& body) -> decltype(body()) {
    status s = status::internal_error;
    std::string message;
    try {
        return body();
    } catch (const status_error& e) {
        s = e.code;
        message = e.what();
    } catch (const std::bad_alloc& e) {
        // A plain std::bad_alloc only says "std::bad_alloc"; the Metal
        // device's names the buffer.
        s = status::out_of_memory;
        message = typeid(e) == typeid(std::bad_alloc) ? "out of memory" : e.what();
    } catch (const std::invalid_argument& e) {
        s = creating ? status::invalid_operator : status::invalid_argument;
        message = e.what();
    } catch (const std::exception& e) {
        message = e.what();
    } catch (...) {
        message = "unknown C++ exception";
    }
    throw std::runtime_error(std::string(status_name(s)) + ": " + message);
}

/// Lets other Julia threads collect garbage while native code runs (what
/// `@ccall gc_safe = true` does on Julia 1.12 and later). Constructed after
/// CxxWrap has converted the arguments; inside it only C++ memory and the raw
/// data of arrays the Julia caller roots are touched. Left on unwind, before
/// CxxWrap rethrows into Julia.
class gc_safe_region {
public:
    gc_safe_region() : ptls_(jl_current_task->ptls), state_(jl_gc_safe_enter(ptls_)) {}
    ~gc_safe_region() { jl_gc_safe_leave(ptls_, state_); }
    gc_safe_region(const gc_safe_region&) = delete;
    gc_safe_region& operator=(const gc_safe_region&) = delete;
private:
    jl_ptls_t ptls_;
    int8_t state_;
};

// The codes src/native.jl passes.
constexpr std::int32_t kBackendCpu = 0, kBackendMetal = 1, kBackendAuto = 2;
constexpr std::int32_t kSamplerGks = 0, kSamplerTraceCycle = 1;
constexpr const char* kPartitionerNames[] = {"block_greedy", "priority_greedy", "baumann_kyng"};
constexpr std::int32_t kStorageAuto = 0, kStorageFloat16 = 16, kStorageFloat32 = 32;

/// The options of one handle, as Julia passes them; default_options() holds
/// apxchol::solve_options' defaults.
struct options {
    std::int32_t backend = kBackendAuto;
    double tol = 0.0;
    std::int32_t max_iter = 0;
    std::uint32_t seed = 0;
    std::int32_t threads = 0;
    std::int32_t sampler = kSamplerGks;
    std::int32_t partitioner = 0;
    std::int32_t keep_factor_values = 0;
    double degree_quantile = 0.0;
    std::uint64_t exact_clique_max_degree = 0;
    std::uint64_t omp_threshold = 0;
    std::int32_t factor_storage = kStorageAuto;
};

options default_options() {
    const apxchol::solve_options so;
    options o;
    o.backend = so.backend == apxchol::solve_backend::cpu   ? kBackendCpu
              : so.backend == apxchol::solve_backend::gpu ? kBackendMetal
                                                          : kBackendAuto;
    o.tol = so.tol;
    o.max_iter = so.max_iter;
    o.seed = so.factor_opts.seed;
    o.threads = so.num_threads;
    o.sampler = so.factor_opts.sampler == apxchol::clique_sampler::trace_cycle
        ? kSamplerTraceCycle : kSamplerGks;
    for (std::int32_t p = 0; p < 3; ++p)
        if (so.factor_opts.is_select == kPartitionerNames[p]) o.partitioner = p;
    o.keep_factor_values = so.keep_factor_values ? 1 : 0;
    o.degree_quantile = so.factor_opts.partition.degree_quantile;
    o.exact_clique_max_degree = so.factor_opts.exact_clique_max_degree;
    o.omp_threshold = so.factor_opts.omp_threshold;
    return o;
}

/// Validates every option before any work, so a later std::invalid_argument
/// during create can only come from the operator contract.
apxchol::solve_options to_solve_options(const options& o) {
    require(o.backend == kBackendCpu || o.backend == kBackendMetal || o.backend == kBackendAuto,
            "backend must be cpu, metal or auto");
    require(o.sampler == kSamplerGks || o.sampler == kSamplerTraceCycle,
            "sampler must be gks or trace_cycle");
    require(o.partitioner >= 0 && o.partitioner < 3, "partitioner is not a known partitioner");
    require(o.keep_factor_values == 0 || o.keep_factor_values == 1, "keep_factor must be a Bool");
    require(o.factor_storage == kStorageAuto || o.factor_storage == kStorageFloat16 ||
                o.factor_storage == kStorageFloat32,
            "factor_storage must be automatic, Float16 or Float32");
    apxchol::solve_options so;
    so.backend = o.backend == kBackendCpu     ? apxchol::solve_backend::cpu
               : o.backend == kBackendMetal ? apxchol::solve_backend::gpu
                                            : apxchol::solve_backend::automatic;
    so.tol = o.tol;
    so.max_iter = o.max_iter;
    so.num_threads = o.threads;
    so.keep_factor_values = o.keep_factor_values == 1;
    so.collect_timings = false;
    so.factor_opts.seed = o.seed;
    so.factor_opts.is_select = kPartitionerNames[o.partitioner];
    so.factor_opts.sampler = o.sampler == kSamplerTraceCycle
        ? apxchol::clique_sampler::trace_cycle : apxchol::clique_sampler::gks;
    so.factor_opts.partition.degree_quantile = o.degree_quantile;
    so.factor_opts.exact_clique_max_degree = static_cast<std::size_t>(o.exact_clique_max_degree);
    so.factor_opts.omp_threshold = static_cast<std::size_t>(o.omp_threshold);
    so.factor_opts.factor_storage = o.factor_storage == kStorageAuto
        ? apxchol::factor_storage_type::automatic
        : o.factor_storage == kStorageFloat16 ? apxchol::factor_storage_type::fp16
                                              : apxchol::factor_storage_type::fp32;
    try {
        apxchol::detail::validate_solve_options(so);
    } catch (const std::invalid_argument& e) {
        fail(status::invalid_argument, e.what());
    }
    return so;
}

/// UNSUPPORTED unless this build has the backend and (for Metal) a device.
void require_backend(std::int32_t backend) {
    if (backend != kBackendMetal) return;
#if defined(APXCHOL_USE_METAL)
    if (!apxchol::backend_available(apxchol::solve_backend::gpu))
        fail(status::unsupported, "no usable Metal device for backend = :metal");
#else
    fail(status::unsupported,
         "this apxchol build has no Metal backend (configure with -DAPXCHOL_DEVICE=METAL)");
#endif
}

/// The route of a new solver. Automatic follows apxchol::solver: the CPU when
/// adopting a host factor, otherwise detail::select_solve_backend in Metal
/// builds; what an explicit Metal request cannot take is UNSUPPORTED.
apxchol::solve_backend resolve_route(std::int32_t backend, const apxchol::solve_options& so,
                                     const Eigen::SparseMatrix<double>& A, bool adopt) {
    if (backend == kBackendCpu) return apxchol::solve_backend::cpu;
#if defined(APXCHOL_USE_METAL)
    if (!adopt) {
        try {
            return apxchol::detail::select_solve_backend(so, &A);
        } catch (const std::invalid_argument& error) {
            if (backend == kBackendMetal) fail(status::unsupported, error.what());
            throw;
        }
    }
#else
    (void)so;
    (void)A;
#endif
    (void)adopt;
    return backend == kBackendMetal ? apxchol::solve_backend::gpu : apxchol::solve_backend::cpu;
}

/// FP16 storage on a CPU route without FP16 support is UNSUPPORTED; it would
/// otherwise surface after the factorization, as another status.
void require_storage(const apxchol::solve_options& so, apxchol::solve_backend route) {
    if (route == apxchol::solve_backend::cpu &&
        so.factor_opts.factor_storage == apxchol::factor_storage_type::fp16 &&
        !apxchol::omp_sptrsv::fp16_supported())
        fail(status::unsupported,
             "FP16 factor storage on the CPU route requires AVX/F16C or Apple ARM64");
}

Eigen::SparseMatrix<double> import_csc(std::int64_t n, const std::int64_t* colptr,
                                       const std::int64_t* rowval, const double* nzval) {
    require(n >= 1, "n must be in [1, 2^31-1]");
    try {
        return apxchol::detail::import_csc(n, colptr, rowval, nzval, 1);
    } catch (const std::invalid_argument& error) {
        fail(status::invalid_argument, error.what());
    }
}

/// Rejects non-finite entries of a caller array of `count` values.
void require_finite(const double* source, Eigen::Index count, const char* what) {
    if (!Eigen::Map<const Eigen::VectorXd>(source, count).allFinite())
        fail(status::invalid_argument, std::string(what) + " contains a non-finite value");
}

/// Copies a vector argument into reusable storage (which makes aliasing with
/// the output safe) and rejects non-finite entries.
void load_vector(const double* source, Eigen::Index n, Eigen::VectorXd& target,
                 const char* what) {
    require_finite(source, n, what);
    target = Eigen::Map<const Eigen::VectorXd>(source, n);
}

/// The operator contract, then whether the factor can precondition the
/// operator. apxchol::solver checks both too, but as std::invalid_argument; a
/// Laplacian factor (rank n-1, centred solutions) on an SDDM operator is an
/// argument error, not an operator error. The test is apxchol::solver's own,
/// on the matrix the factorization would see, so a factor never refuses its
/// own operator.
void check_adoptable(const apxchol::factorization& F, const Eigen::SparseMatrix<double>& A,
                     const apxchol::solve_options& so) {
    apxchol::require_operator(A, so.factor_opts.lump_max_mass);
    if (F.sddm) return;
    for (Eigen::Index col = 0; col < A.cols(); ++col) {
        double diagonal = 0, positive = 0, degree = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator entry(A, col); entry; ++entry)
            if (entry.row() == entry.col()) diagonal = entry.value();
            else if (entry.value() > 0) positive += entry.value();
            else degree -= entry.value();
        diagonal += positive;
        if (diagonal - degree > diagonal * 1e-12)
            fail(status::invalid_argument,
                 "a Laplacian factor (rank n-1, centred solutions) cannot precondition an SDDM "
                 "operator");
    }
}

bool has_values(const apxchol::factorization& F) {
    const auto nnz = static_cast<std::size_t>(F.L.nonZeros());
    return F.L.inner_.size() == nnz && F.L.vals_.size() == nnz;
}

constexpr const char* kReleasedValues =
    "the factor values were released after setup; create the solver with keep_factor=true";

/// L (CSC, permuted space, 1-based) and perm (perm[v] = v's elimination
/// position, 1-based). With colptr NULL only perm is written.
void export_arrays(const apxchol::factorization& F, std::size_t n, std::int64_t* colptr,
                   std::int64_t* rowval, float* nzval, std::int64_t* perm) {
    if (F.perm.size() != n || static_cast<std::size_t>(F.L.rows()) != n)
        fail(status::internal_error, "factor dimension differs from the operator");
    if (colptr != nullptr) {
        if (!has_values(F)) fail(status::no_factor_values, kReleasedValues);
        const auto nnz = static_cast<std::size_t>(F.L.nonZeros());
        for (std::size_t j = 0; j <= n; ++j) colptr[j] = static_cast<std::int64_t>(F.L.outer_[j]) + 1;
        for (std::size_t p = 0; p < nnz; ++p) {
            rowval[p] = static_cast<std::int64_t>(F.L.inner_[p]) + 1;
            nzval[p] = F.L.vals_[p];
        }
    }
    for (std::size_t v = 0; v < n; ++v) perm[v] = static_cast<std::int64_t>(F.perm[v]) + 1;
}

// (backend, n, factor_nnz, lumped_offdiag, rounds, peak_graph_bytes,
//  setup_seconds, sddm, setup_max_threads, factor_storage)
using stats_tuple = std::tuple<std::int32_t, std::int64_t, std::int64_t, std::int64_t,
                               std::int64_t, std::int64_t, double, std::int32_t, std::int32_t,
                               std::int32_t>;

stats_tuple make_stats(const apxchol::factorization& F, std::int32_t backend, double setup_seconds,
                       std::int32_t setup_max_threads, std::int32_t storage) {
    return {backend, static_cast<std::int64_t>(F.perm.size()),
            static_cast<std::int64_t>(F.L.nonZeros()), static_cast<std::int64_t>(F.lumped_offdiag),
            static_cast<std::int64_t>(F.rounds.size()),
            static_cast<std::int64_t>(F.peak_graph_bytes), setup_seconds, F.sddm ? 1 : 0,
            setup_max_threads, storage};
}

/// Immutable after creation: every call on it only reads.
struct Factor {
    apxchol::factorization F;  // values always retained
    double setup_seconds = 0.0;
    std::int32_t setup_max_threads = 1;
};

struct Solver {
    std::int32_t threads = 0;
    std::unique_ptr<apxchol::solver> owner;
    Eigen::VectorXd b_buffer, x0_buffer;
    double setup_seconds = 0.0;
    std::int32_t setup_max_threads = 1;

    const apxchol::factorization& factor() const { return owner->factor(); }
    Eigen::Index n() const { return owner->rows(); }

    // b and x0 are copied: aliased output is safe, and solver::solve reads them
    // as Eigen vectors.
    apxchol::solve_result solve_one(const double* b, const double* x0, double* x, double tol,
                                    int max_iter) {
        load_vector(b, n(), b_buffer, "b");
        if (x0 != nullptr) load_vector(x0, n(), x0_buffer, "x0");
        Eigen::Map<Eigen::VectorXd> out(x, n());
        try {
            return owner->solve(b_buffer, out, tol, max_iter, x0 != nullptr ? &x0_buffer : nullptr);
        } catch (const std::overflow_error& e) {
            fail(status::unsupported, e.what());
        } catch (const std::underflow_error& e) {
            fail(status::unsupported, e.what());
        }
    }
};

/// A new solver factorizing A (factor NULL) or adopting a copy of *factor.
Solver* create_solver(std::int64_t n, const std::int64_t* colptr, const std::int64_t* rowval,
                      const double* nzval, const Factor* factor, const options& opt) {
    const bool adopt = factor != nullptr;
    apxchol::solve_options so = to_solve_options(opt);
    require_backend(opt.backend);
    const Eigen::SparseMatrix<double> A = import_csc(n, colptr, rowval, nzval);
    if (adopt)
        require(static_cast<std::size_t>(A.rows()) == factor->F.perm.size(),
                "the factor's dimension differs from the operator's");

    auto solver = std::make_unique<Solver>();
    solver->threads = opt.threads;
    const apxchol::detail::thread_budget scope(opt.threads);
    solver->setup_max_threads = omp_get_max_threads();
    const auto start = clock_type::now();
    // Resolved once here; the owner then runs exactly this route.
    so.backend = resolve_route(opt.backend, so, A, adopt);
    const bool gpu = so.backend == apxchol::solve_backend::gpu;
    require_storage(so, so.backend);
    if (adopt) check_adoptable(factor->F, A, so);
    try {
        solver->owner = adopt ? std::make_unique<apxchol::solver>(A, factor->F, so)
                              : std::make_unique<apxchol::solver>(A, so);
    } catch (const std::domain_error& e) {
        if (gpu) fail(status::unsupported, e.what());
        throw;
    } catch (const std::length_error& e) {
        if (gpu) fail(status::unsupported, e.what());
        throw;
    }
    solver->setup_seconds = seconds_since(start);
    return solver.release();
}

/// k columns of n values each; per column the iterations, the original
/// system's relative residual and convergence. Every column is validated
/// before the first is solved. Returns the elapsed seconds.
double solve(Solver& s, std::int64_t k, const double* b, const double* x0, double* x, double tol,
             std::int32_t max_iter, std::int64_t* iterations, double* residuals,
             std::int32_t* converged) {
    const auto start = clock_type::now();
    if (k == 0) return 0.0;
    const std::int64_t n = s.n();
    require(k > 0 && k <= PTRDIFF_MAX / static_cast<std::int64_t>(sizeof(double)) / n,
            "n * k * sizeof(double) overflows");
    require_finite(b, n * k, "b");
    if (x0 != nullptr) require_finite(x0, n * k, "x0");
    const apxchol::detail::thread_budget scope(s.threads);
    for (std::int64_t c = 0; c < k; ++c) {
        const std::int64_t off = c * n;
        const apxchol::solve_result r =
            s.solve_one(b + off, x0 != nullptr ? x0 + off : nullptr, x + off, tol, max_iter);
        iterations[c] = static_cast<std::int64_t>(r.iterations);
        residuals[c] = r.residual;
        converged[c] = r.converged ? 1 : 0;
    }
    return seconds_since(start);
}

void apply(Solver& s, const double* r, double* z) {
    const apxchol::detail::thread_budget scope(s.threads);
    const Eigen::Index n = s.n();
    const double* input = r;
    if (r == z) {
        load_vector(r, n, s.b_buffer, "r");
        input = s.b_buffer.data();
    } else {
        require_finite(r, n, "r");
    }
    const Eigen::Map<const Eigen::MatrixXd> R(input, n, 1);
    Eigen::Map<Eigen::MatrixXd> Z(z, n, 1);
    s.owner->apply(apxchol::const_matrix_view(R), apxchol::matrix_view(Z));
}

Factor* create_factor(std::int64_t n, const std::int64_t* colptr, const std::int64_t* rowval,
                      const double* nzval, const options& opt) {
    const apxchol::solve_options so = to_solve_options(opt);
    const Eigen::SparseMatrix<double> A = import_csc(n, colptr, rowval, nzval);
    auto factor = std::make_unique<Factor>();
    const apxchol::detail::thread_budget scope(opt.threads);
    factor->setup_max_threads = omp_get_max_threads();
    const auto start = clock_type::now();
    // Public factorization retains native FP32 values on every build.
    factor->F = apxchol::factorize(A, so.factor_opts);
    factor->setup_seconds = seconds_since(start);
    return factor.release();
}

Factor* copy_factor(const Solver& s) {
    const apxchol::factorization& F = s.factor();
    if (!has_values(F)) fail(status::no_factor_values, kReleasedValues);
    auto factor = std::make_unique<Factor>();
    factor->F = F;
    factor->setup_seconds = s.setup_seconds;
    factor->setup_max_threads = s.setup_max_threads;
    return factor.release();
}

}  // namespace

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.add_type<Solver>("Solver");
    mod.add_type<Factor>("Factor");

    mod.method("version", []() { return std::string(APXCHOL_VERSION "+" APXCHOL_GIT_SHA); });
    mod.method("max_threads", []() { return static_cast<std::int32_t>(omp_get_max_threads()); });
    mod.method("backend_available", [](std::int32_t backend) {
        if (backend == kBackendCpu || backend == kBackendAuto) return true;
#if defined(APXCHOL_USE_METAL)
        if (backend == kBackendMetal)
            return apxchol::backend_available(apxchol::solve_backend::gpu);
#endif
        return false;
    });
    // (backend, tol, max_iter, seed, threads, sampler, partitioner,
    //  keep_factor_values, degree_quantile, exact_clique_max_degree,
    //  omp_threshold, factor_storage): the order create_solver takes them in.
    mod.method("default_options", []() {
        const options o = default_options();
        return std::make_tuple(o.backend, o.tol, o.max_iter, o.seed, o.threads, o.sampler,
                               o.partitioner, o.keep_factor_values, o.degree_quantile,
                               o.exact_clique_max_degree, o.omp_threshold, o.factor_storage);
    });

    mod.method("create_solver",
               [](std::int64_t n, const std::int64_t* colptr, const std::int64_t* rowval,
                  const double* nzval, const Factor* factor, std::int32_t backend, double tol,
                  std::int32_t max_iter, std::uint32_t seed, std::int32_t threads,
                  std::int32_t sampler, std::int32_t partitioner, std::int32_t keep_factor_values,
                  double degree_quantile, std::uint64_t exact_clique_max_degree,
                  std::uint64_t omp_threshold, std::int32_t factor_storage) {
        const gc_safe_region region;
        return guarded(true, [&] {
            return create_solver(n, colptr, rowval, nzval, factor,
                                 {backend, tol, max_iter, seed, threads, sampler, partitioner,
                                  keep_factor_values, degree_quantile, exact_clique_max_degree,
                                  omp_threshold, factor_storage});
        });
    });
    // The factorization fields only; a factor always keeps its values.
    mod.method("create_factor",
               [](std::int64_t n, const std::int64_t* colptr, const std::int64_t* rowval,
                  const double* nzval, std::uint32_t seed, std::int32_t threads,
                  std::int32_t sampler, std::int32_t partitioner, double degree_quantile,
                  std::uint64_t exact_clique_max_degree, std::uint64_t omp_threshold) {
        const gc_safe_region region;
        return guarded(true, [&] {
            options o = default_options();
            o.seed = seed;
            o.threads = threads;
            o.sampler = sampler;
            o.partitioner = partitioner;
            o.degree_quantile = degree_quantile;
            o.exact_clique_max_degree = exact_clique_max_degree;
            o.omp_threshold = omp_threshold;
            return create_factor(n, colptr, rowval, nzval, o);
        });
    });
    mod.method("copy_factor", [](const Solver* s) {
        const gc_safe_region region;
        return guarded(false, [&] { return copy_factor(*s); });
    });
    mod.method("destroy_solver", [](Solver* s) { delete s; });
    mod.method("destroy_factor", [](Factor* f) { delete f; });

    mod.method("solve", [](Solver* s, std::int64_t k, const double* b, const double* x0, double* x,
                           double tol, std::int32_t max_iter, std::int64_t* iterations,
                           double* residuals, std::int32_t* converged) {
        const gc_safe_region region;
        return guarded(false, [&] {
            return solve(*s, k, b, x0, x, tol, max_iter, iterations, residuals, converged);
        });
    });
    mod.method("apply", [](Solver* s, const double* r, double* z) {
        const gc_safe_region region;
        guarded(false, [&] { apply(*s, r, z); });
    });

    mod.method("solver_stats", [](const Solver* s) {
        return guarded(false, [&] {
            const std::int32_t storage =
                s->owner->factor_storage() == apxchol::factor_storage_type::fp16
                    ? kStorageFloat16 : kStorageFloat32;
            return make_stats(s->factor(),
                              s->owner->backend() == apxchol::solve_backend::cpu ? kBackendCpu
                                                                                 : kBackendMetal,
                              s->setup_seconds, s->setup_max_threads, storage);
        });
    });
    mod.method("factor_stats", [](const Factor* f) {
        return guarded(false, [&] {
            return make_stats(f->F, kBackendCpu, f->setup_seconds, f->setup_max_threads,
                              kStorageFloat32);
        });
    });
    mod.method("export_solver_factor", [](const Solver* s, std::int64_t* colptr, std::int64_t* rowval,
                                   float* nzval, std::int64_t* perm) {
        const gc_safe_region region;
        guarded(false, [&] {
            export_arrays(s->factor(), static_cast<std::size_t>(s->n()), colptr, rowval, nzval, perm);
        });
    });
    mod.method("export_factor", [](const Factor* f, std::int64_t* colptr, std::int64_t* rowval,
                                   float* nzval, std::int64_t* perm) {
        const gc_safe_region region;
        guarded(false, [&] { export_arrays(f->F, f->F.perm.size(), colptr, rowval, nzval, perm); });
    });
}
