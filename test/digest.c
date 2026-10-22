/* Reads the system test/digest.jl writes, solves it through apxchol's C ABI
 * with the options the Julia test uses, and prints FNV-1a digests of the
 * exported factor and of the solution, so the Julia binding can be compared
 * with C byte for byte on the same library. */
#include "apxchol/c_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv1a(const void* data, size_t bytes) {
    const unsigned char* p = data;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}

static void* read_exact(FILE* f, size_t bytes) {
    void* p = malloc(bytes > 0 ? bytes : 1);
    if (p == NULL || fread(p, 1, bytes, f) != bytes) {
        fprintf(stderr, "digest: short read\n");
        exit(2);
    }
    return p;
}

static void print(const char* name, const void* data, size_t bytes) {
    printf("%s %016llx\n", name, (unsigned long long)fnv1a(data, bytes));
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    FILE* f = fopen(argv[1], "rb");
    if (f == NULL) return 2;
    int64_t* header = read_exact(f, 2 * sizeof(int64_t));
    const int64_t n = header[0], nnz = header[1];
    int64_t* colptr = read_exact(f, (size_t)(n + 1) * sizeof(int64_t));
    int64_t* rowval = read_exact(f, (size_t)nnz * sizeof(int64_t));
    double* nzval = read_exact(f, (size_t)nnz * sizeof(double));
    double* b = read_exact(f, (size_t)n * sizeof(double));
    fclose(f);

    apxchol_options o;
    char error[512];
    if (apxchol_options_default(&o, sizeof o) != APXCHOL_STATUS_SUCCESS) return 1;
    o.backend = APXCHOL_BACKEND_CPU;
    o.threads = 1;
    o.seed = 7;
    o.keep_factor_values = 1;
    o.factor_storage = APXCHOL_FACTOR_STORAGE_FLOAT32;
    apxchol_solver* s = NULL;
    if (apxchol_solver_create(n, colptr, rowval, nzval, 1, &o, &s, error, sizeof error) !=
        APXCHOL_STATUS_SUCCESS) {
        fprintf(stderr, "digest: %s\n", error);
        return 1;
    }
    double* x = malloc((size_t)n * sizeof(double));
    apxchol_solve_info info;
    info.struct_size = sizeof info;
    const apxchol_status solved =
        apxchol_solver_solve(s, b, NULL, x, -1.0, -1, &info, error, sizeof error);
    apxchol_stats stats;
    stats.struct_size = sizeof stats;
    if ((solved != APXCHOL_STATUS_SUCCESS && solved != APXCHOL_STATUS_NOT_CONVERGED) ||
        apxchol_solver_stats(s, &stats) != APXCHOL_STATUS_SUCCESS) {
        fprintf(stderr, "digest: %s\n", error);
        return 1;
    }
    int64_t* lcolptr = malloc((size_t)(n + 1) * sizeof(int64_t));
    int64_t* lrowval = malloc((size_t)stats.factor_nnz * sizeof(int64_t));
    float* lnzval = malloc((size_t)stats.factor_nnz * sizeof(float));
    int64_t* perm = malloc((size_t)n * sizeof(int64_t));
    if (apxchol_solver_export_factor(s, 1, lcolptr, lrowval, lnzval, perm, error, sizeof error) !=
        APXCHOL_STATUS_SUCCESS) {
        fprintf(stderr, "digest: %s\n", error);
        return 1;
    }
    uint64_t residual;
    memcpy(&residual, &info.relative_residual, sizeof residual);
    print("colptr", lcolptr, (size_t)(n + 1) * sizeof(int64_t));
    print("rowval", lrowval, (size_t)stats.factor_nnz * sizeof(int64_t));
    print("nzval", lnzval, (size_t)stats.factor_nnz * sizeof(float));
    print("perm", perm, (size_t)n * sizeof(int64_t));
    print("x", x, (size_t)n * sizeof(double));
    printf("iterations %lld\nresidual %016llx\n", (long long)info.iterations,
           (unsigned long long)residual);
    apxchol_solver_destroy(s);
    return 0;
}
