cmake_minimum_required(VERSION 3.16)
project(sampler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

set(SAMPLER_SOURCES
    src/lapack_wrapper.cpp
    src/interp_decomp.cpp
)
set(SAMPLER_INCLUDE include)
set(SAMPLER_LIBS "${TORCH_LIBRARIES}" lapack)

add_executable(sampler src/main.cpp ${SAMPLER_SOURCES})
target_include_directories(sampler PRIVATE ${SAMPLER_INCLUDE})
target_link_libraries(sampler ${SAMPLER_LIBS})

# Tests
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(test_interp_decomp tests/test_interp_decomp.cpp ${SAMPLER_SOURCES})
target_include_directories(test_interp_decomp PRIVATE ${SAMPLER_INCLUDE})
target_link_libraries(test_interp_decomp ${SAMPLER_LIBS} Catch2::Catch2WithMain)

add_executable(test_tt
    tests/test_tt.cpp
    tests/test_tt_id.cpp
    ${SAMPLER_SOURCES}
)
target_include_directories(test_tt PRIVATE ${SAMPLER_INCLUDE})
target_link_libraries(test_tt ${SAMPLER_LIBS} Catch2::Catch2WithMain)

add_executable(test_q_tt_id
    tests/test_q_tt_id.cpp
    ${SAMPLER_SOURCES}
)
target_include_directories(test_q_tt_id PRIVATE ${SAMPLER_INCLUDE})
target_link_libraries(test_q_tt_id ${SAMPLER_LIBS} Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(test_interp_decomp)
catch_discover_tests(test_tt)
catch_discover_tests(test_q_tt_id)
