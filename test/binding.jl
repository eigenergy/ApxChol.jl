@testset "CxxWrap module and build checks" begin
    @test ApxChol.STATUS == (:invalid_argument, :invalid_operator, :no_factor_values,
                             :unsupported, :out_of_memory, :internal_error)
    @test ApxChol.BACKENDS == (cpu=0, metal=1, auto=2)
    @test ApxChol.SAMPLERS == (gks=0, trace_cycle=1)
    @test ApxChol.PARTITIONERS == (block_greedy=0, priority_greedy=1, baumann_kyng=2)
    @test map(ApxChol.storage_code, (nothing, Float16, Float32)) == (0, 16, 32)
    @test map(ApxChol.storage_type, (0, 16, 32)) == (nothing, Float16, Float32)

    # apxchol::solve_options' defaults, in the order create_solver takes them.
    d = ApxChol.DEFAULTS[]
    @test (d.backend, d.tol, d.max_iter, d.seed, d.threads, d.sampler, d.partitioner,
           d.keep_factor_values, d.degree_quantile, d.exact_clique_max_degree, d.omp_threshold,
           d.factor_storage) == (2, 1e-8, 200, 42, 0, 0, 0, 0, -1.0, 0, 2000, 0)
    @test length(ApxChol.Native.default_options()) == fieldcount(ApxChol.Options)

    # Errors carry their status as a message prefix; CxxWrap's own pass through.
    e = ApxChol.native_error(ErrorException("invalid_operator: not symmetric"))
    @test e isa ApxCholError && e.status == :invalid_operator && e.message == "not symmetric"
    other = ErrorException("C++ object of type Solver was deleted")
    @test ApxChol.native_error(other) === other
    @test ApxChol.native_error(ErrorException("success: no")) isa ErrorException

    info = ApxChol.build_info()
    @test endswith(ApxChol.version(), "+" * first(info["apxchol_revision"], 7))
    @test info["apxchol_source"] != "pin" || info["apxchol_revision"] == ApxChol.pin_revision()
    @test info["julia"] == "$(VERSION.major).$(VERSION.minor)"
    @test ApxChol.check_build() === nothing
    @test ApxChol.openmp_enabled()
    @test ApxChol.backend_available(:cpu) === true && ApxChol.backend_available(:auto) === true
    @test ApxChol.backend_available(:metal) isa Bool
    @test !ApxChol.backend_available(:cuda)
    if VERSION >= v"1.11"
        @test all(s -> Base.ispublic(ApxChol, s),
                  (:solve, :solve!, :apply!, :factor, :ordering, :stats, :options, :version))
        @test !Base.isexported(ApxChol, :solve)
    end
end
