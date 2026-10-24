@testset "Aqua" begin
    # Not recursive: ApxChol.Native holds only CxxWrap's generated methods, whose
    # __delete methods are ambiguous with those of CxxWrap's standard library
    # and take minutes to enumerate.
    Aqua.test_all(ApxChol; ambiguities=(recursive=false,))
end
