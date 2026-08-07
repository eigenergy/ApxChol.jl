cmake_minimum_required(VERSION 3.20)
project(ApproxCholeskyBridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT APXCHOL_SOURCE_DIR)
    message(FATAL_ERROR "APXCHOL_SOURCE_DIR must point to an apxchol checkout")
endif()

include(FetchContent)
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(eigen)
endif()

add_library(approxcholesky SHARED
    ${APXCHOL_SOURCE_DIR}/src/c_api.cpp
    ${APXCHOL_SOURCE_DIR}/src/factorization.cpp
    ${APXCHOL_SOURCE_DIR}/src/solve.cpp)
target_include_directories(approxcholesky PRIVATE ${APXCHOL_SOURCE_DIR}/include)
target_link_libraries(approxcholesky PRIVATE Eigen3::Eigen)
target_compile_definitions(approxcholesky PRIVATE
    APXCHOL_SPTRSV_FP32
    APXCHOL_POOL_FP32)

include(${APXCHOL_SOURCE_DIR}/cmake/ApxCholOpenMP.cmake)
apxchol_enable_openmp(approxcholesky PRIVATE)

if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(approxcholesky PRIVATE -O3 -march=native)
endif()

install(TARGETS approxcholesky
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
