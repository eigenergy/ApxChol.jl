cmake_minimum_required(VERSION 3.20)
project(ApxCholDigest LANGUAGES CXX)
# apxchol's installed core in deps/usr, as any C++ consumer finds it.
find_package(apxchol CONFIG REQUIRED)
add_executable(digest digest.cpp)
target_link_libraries(digest PRIVATE apxchol::core)
