@testset "C ABI mirror and handshake" begin
    offsets(T) = [fieldoffset(T, i) for i in 1:fieldcount(T)]
    @test sizeof(ApxChol.COptions) == 80
    @test offsets(ApxChol.COptions) == [0, 4, 8, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72]
    @test sizeof(ApxChol.CSolveInfo) == 32
    @test offsets(ApxChol.CSolveInfo) == [0, 4, 8, 16, 24]
    @test sizeof(ApxChol.CStats) == 72
    @test offsets(ApxChol.CStats) == [0, 4, 8, 16, 24, 32, 40, 48, 56, 60, 64]
    @test ApxChol.STATUS == (:success, :not_converged, :invalid_argument, :invalid_operator,
                             :no_factor_values, :unsupported, :out_of_memory, :internal_error)
    @test ApxChol.BACKENDS == (cpu=0, metal=1, auto=2)
    @test ApxChol.SAMPLERS == (gks=0, trace_cycle=1)
    @test ApxChol.PARTITIONERS == (block_greedy=0, priority_greedy=1, baumann_kyng=2)
    @test map(ApxChol.storage_code, (nothing, Float16, Float32)) == (0, 16, 32)
    @test map(ApxChol.storage_type, (0, 16, 32)) == (nothing, Float16, Float32)

    # The library takes exactly this options size: smaller is an argument
    # error, larger unsupported (the handshake __init__ relies on).
    for (size, status) in ((sizeof(ApxChol.COptions), 0), (72, 2), (88, 5))
        buffer = zeros(UInt8, 96)
        @test (@ccall ApxChol.libapxchol.apxchol_options_default(
            buffer::Ptr{UInt8}, size::Csize_t)::Int32) == status
    end
    d = ApxChol.DEFAULTS[]
    @test d.struct_size == 80 && d.backend == 2 && d.factor_storage == 0

    info = ApxChol.build_info()
    @test endswith(ApxChol.version(), "+" * first(info["apxchol_revision"], 7))
    @test info["apxchol_source"] != "pin" || info["apxchol_revision"] == ApxChol.pin_revision()
    @test ApxChol.openmp_enabled()
    @test ApxChol.backend_available(:cpu) && ApxChol.backend_available(:auto)
    @test !ApxChol.backend_available(:cuda)
    if VERSION >= v"1.11"
        @test all(s -> Base.ispublic(ApxChol, s),
                  (:solve, :solve!, :apply!, :factor, :ordering, :stats, :options, :version))
        @test !Base.isexported(ApxChol, :solve)
    end
end
