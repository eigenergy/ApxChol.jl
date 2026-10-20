cmake_minimum_required(VERSION 3.20)
project(ApxCholJL LANGUAGES CXX)

# libapxchol: apxchol's C ABI over its core, configured by upstream's
# cmake/apxchol_core.cmake exactly as apxchol's Python bindings are, so the
# device (APXCHOL_DEVICE), sources, Eigen and OpenMP settings follow upstream.
# deps/build.jl supplies every input, including LLVMOpenMP_jll as the OpenMP.
if(NOT APXCHOL_SOURCE_DIR)
    message(FATAL_ERROR "APXCHOL_SOURCE_DIR must be set (deps/build.jl passes it)")
endif()
include("${APXCHOL_SOURCE_DIR}/cmake/apxchol_core.cmake")

# apxchol/version.h, as apxchol's root project generates it.
file(STRINGS "${APXCHOL_SOURCE_DIR}/CMakeLists.txt" _project REGEX "^project\\(apxchol VERSION ")
string(REGEX REPLACE "^project\\(apxchol VERSION ([0-9.]+).*" "\\1" APXCHOL_VERSION "${_project}")
if(NOT APXCHOL_GIT_SHA)
    set(APXCHOL_GIT_SHA "unknown")
endif()
configure_file("${APXCHOL_SOURCE_DIR}/include/apxchol/version.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/apxchol/version.h" @ONLY)

add_library(apxchol SHARED "${APXCHOL_SOURCE_DIR}/src/c_api.cpp")
apxchol_configure_core(apxchol)
# Only the C ABI (visibility("default") in c_api.h) is exported; the library
# finds the JLL's libomp through the directory it was linked from.
set_target_properties(apxchol PROPERTIES
    CXX_VISIBILITY_PRESET hidden OBJCXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    INSTALL_RPATH_USE_LINK_PATH ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(apxchol PRIVATE "LINKER:--no-undefined")
endif()

install(TARGETS apxchol LIBRARY DESTINATION lib)
install(FILES "${APXCHOL_SOURCE_DIR}/LICENSE" DESTINATION share/licenses/apxchol)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md"
        DESTINATION share/licenses/ApxChol)
