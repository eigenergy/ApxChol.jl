// Reads the system test/digest.jl writes, solves it with apxchol::solver (the
// installed C++ core, without ApxChol.jl's module) and the options the Julia
// test uses, and prints FNV-1a digests of the factor and of the solution, so
// the Julia binding can be compared with apxchol's own C++ byte for byte.
#include <apxchol/detail/csc_import.h>
#include <apxchol/solver/solve.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

std::uint64_t fnv1a(const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = 14695981039346656037ULL;
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}

template <class T>
std::vector<T> read(std::ifstream& in, std::int64_t count) {
    std::vector<T> v(static_cast<std::size_t>(count));
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(T)));
    return v;
}

template <class T>
void print_digest(const char* name, const std::vector<T>& v) {
    std::printf("%s %016llx\n", name, static_cast<unsigned long long>(fnv1a(v.data(), v.size() * sizeof(T))));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream in(argv[1], std::ios::binary);
    const auto header = read<std::int64_t>(in, 2);
    const std::int64_t n = header[0], nnz = header[1];
    const auto colptr = read<std::int64_t>(in, n + 1);
    const auto rowval = read<std::int64_t>(in, nnz);
    const auto nzval = read<double>(in, nnz);
    const auto b = read<double>(in, n);
    if (!in) return 2;

    apxchol::solve_options so;
    so.backend = apxchol::solve_backend::cpu;
    so.num_threads = 1;
    so.factor_opts.seed = 7;
    so.keep_factor_values = true;
    so.factor_opts.factor_storage = apxchol::factor_storage_type::fp32;
    so.collect_timings = false;
    const apxchol::solver s(apxchol::detail::import_csc(n, colptr.data(), rowval.data(), nzval.data(), 1), so);
    const apxchol::solve_result r = s.solve(Eigen::Map<const Eigen::VectorXd>(b.data(), n));

    // 1-based, as ApxChol.factor exports them.
    const apxchol::factorization& F = s.factor();
    std::vector<std::int64_t> lcolptr, lrowval, perm;
    for (auto p : F.L.outer_) lcolptr.push_back(static_cast<std::int64_t>(p) + 1);
    for (auto i : F.L.inner_) lrowval.push_back(static_cast<std::int64_t>(i) + 1);
    for (auto v : F.perm) perm.push_back(static_cast<std::int64_t>(v) + 1);
    const std::vector<float> lnzval(F.L.vals_.begin(), F.L.vals_.end());
    const std::vector<double> x(r.x.data(), r.x.data() + r.x.size());
    std::uint64_t residual;
    std::memcpy(&residual, &r.residual, sizeof residual);
    print_digest("colptr", lcolptr);
    print_digest("rowval", lrowval);
    print_digest("nzval", lnzval);
    print_digest("perm", perm);
    print_digest("x", x);
    std::printf("iterations %lld\nresidual %016llx\n", static_cast<long long>(r.iterations),
                static_cast<unsigned long long>(residual));
    return 0;
}
