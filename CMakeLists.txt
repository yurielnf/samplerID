cmake_minimum_required(VERSION 3.16)
project(sampler)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

add_executable(sampler
    main.cpp
    id.cpp
    tt.cpp
    id.h
    tt.h
    tt_ci.h
    tt_ci.cpp
    index_set.h
)
target_link_libraries(sampler "${TORCH_LIBRARIES}" lapack)

# Tests
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(test_id tests/test_id.cpp id.cpp)
target_link_libraries(test_id "${TORCH_LIBRARIES}" lapack Catch2::Catch2WithMain)

add_executable(test_tt tests/test_tt.cpp tt.cpp id.cpp
    tests/test_tt_ci.cpp)
target_link_libraries(test_tt "${TORCH_LIBRARIES}" lapack Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(test_id)
catch_discover_tests(test_tt)
