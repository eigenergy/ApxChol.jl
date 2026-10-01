cmake_minimum_required(VERSION 3.20)
project(ApxCholJL LANGUAGES CXX)

# libapxchol: the apxchol core translation units plus its C ABI, compiled
# directly (as the Python and Octave bindings do) so the root project's
# CLI/test dependencies are never fetched. deps/build.jl supplies every input.

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

foreach(_required APXCHOL_SOURCE_DIR APXCHOL_JL_OPENMP_INCLUDE_DIR APXCHOL_JL_OPENMP_LIBRARY)
    if(NOT ${_required})
        message(FATAL_ERROR "${_required} must be set (deps/build.jl passes it)")
    endif()
endforeach()
set(_src "${APXCHOL_SOURCE_DIR}")

# Eigen 3.4.0, pinned by commit and headers only: never a system Eigen (Homebrew
# ships 5.x) and never Eigen's own CMake project.
include(FetchContent)
FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c # 3.4.0
    SOURCE_SUBDIR headers-only-do-not-configure)
FetchContent_MakeAvailable(eigen)

# apxchol/version.h, as the root project generates it.
file(STRINGS "${_src}/CMakeLists.txt" _apxchol_project REGEX "^project\\(apxchol VERSION ")
string(REGEX REPLACE "^project\\(apxchol VERSION ([0-9.]+).*" "\\1" APXCHOL_VERSION "${_apxchol_project}")
if(NOT APXCHOL_GIT_SHA)
    set(APXCHOL_GIT_SHA "unknown")
endif()
configure_file("${_src}/include/apxchol/version.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/apxchol/version.h" @ONLY)

add_library(apxchol SHARED
    vendor/src/c_api.cpp
    "${_src}/src/factorization.cpp"
    "${_src}/src/operator_class.cpp"
    "${_src}/src/solve.cpp")
target_include_directories(apxchol PRIVATE
    vendor/include "${_src}/include" "${CMAKE_CURRENT_BINARY_DIR}/include")
target_include_directories(apxchol SYSTEM PRIVATE
    "${eigen_SOURCE_DIR}" "${APXCHOL_JL_OPENMP_INCLUDE_DIR}")
target_compile_definitions(apxchol PRIVATE APXCHOL_POOL_FP32)
# Only the C ABI (visibility("default") in c_api.h) is exported.
set_target_properties(apxchol PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

if(APPLE)
    target_compile_options(apxchol PRIVATE
        "SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/compat/apxchol_macos_compat.h")
endif()

# Tune for the build machine (-mcpu=native on arm64), as the root project does.
# Only the C ABI crosses the library boundary, so no Eigen ABI is shared.
include("${_src}/cmake/apxchol_native_arch.cmake")
if(APXCHOL_NATIVE_ARCH_FLAG)
    target_compile_options(apxchol PRIVATE ${APXCHOL_NATIVE_ARCH_FLAG})
endif()
target_compile_options(apxchol PRIVATE -O3)

# LLVMOpenMP_jll is the only OpenMP runtime: the OpenMP flag at compile time
# only, and the JLL's libomp linked explicitly (never libgomp). GCC objects
# resolve through libomp's GOMP entry points.
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(apxchol PRIVATE "SHELL:-Xclang -fopenmp")
else()
    target_compile_options(apxchol PRIVATE -fopenmp)
endif()
find_package(Threads REQUIRED)
target_link_libraries(apxchol PRIVATE "${APXCHOL_JL_OPENMP_LIBRARY}" Threads::Threads)
get_filename_component(_omp_dir "${APXCHOL_JL_OPENMP_LIBRARY}" DIRECTORY)
set_target_properties(apxchol PROPERTIES BUILD_RPATH "${_omp_dir}" INSTALL_RPATH "${_omp_dir}")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(apxchol PRIVATE "LINKER:--no-undefined")
endif()

install(TARGETS apxchol LIBRARY DESTINATION lib)
