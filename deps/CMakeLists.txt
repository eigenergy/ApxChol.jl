cmake_minimum_required(VERSION 3.20)
project(ApxCholJL LANGUAGES CXX)

# libapxchol_jl: the CxxWrap module in apxchol_jl.cpp over apxchol's installed
# C++ core (find_package(apxchol CONFIG), apxchol::core), the route upstream
# documents for installed consumers. deps/build.jl installs that core into
# CMAKE_PREFIX_PATH and supplies every other input, including LLVMOpenMP_jll as
# the OpenMP and CxxWrap's libcxxwrap_julia for the running Julia.
find_package(apxchol CONFIG REQUIRED)
find_package(JlCxx CONFIG REQUIRED)

add_library(apxchol_jl SHARED apxchol_jl.cpp)
target_link_libraries(apxchol_jl PRIVATE apxchol::core JlCxx::cxxwrap_julia)
# Default visibility: CxxWrap's type registry fails to resolve types (even
# void) in a module built with hidden symbols. The library finds the JLL's libomp
# and libcxxwrap_julia through the directories it was linked from.
set_target_properties(apxchol_jl PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(apxchol_jl PRIVATE "LINKER:--no-undefined")
endif()

install(TARGETS apxchol_jl LIBRARY DESTINATION lib)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md"
        DESTINATION share/licenses/ApxChol)
