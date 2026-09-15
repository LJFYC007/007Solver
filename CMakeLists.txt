cmake_minimum_required(VERSION 3.20)
if(CMAKE_HOST_WIN32)
    find_program(CMAKE_CXX_COMPILER cl.exe REQUIRED)
endif()
if(CMAKE_HOST_APPLE)
    # Keep the sidecar's minimum OS aligned with tauri.macos.conf.json.
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
endif()
project(007Solver LANGUAGES CXX)

if(MSVC)
    # Match Release inlining while retaining profiling symbols.
    string(REPLACE "/Ob1" "/Ob2" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)

add_subdirectory(solver)

add_executable(007SolverTests tests/domain_tests.cpp tests/tests.cpp)
target_link_libraries(007SolverTests PRIVATE solver_service_support GTest::gtest_main)
target_include_directories(007SolverTests PRIVATE third_party/nlohmann_json/single_include)
target_compile_definitions(
    007SolverTests
    PRIVATE
        TEST_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures/"
)
if(MSVC)
    target_compile_options(007SolverTests PRIVATE /utf-8)
endif()

enable_testing()
add_test(NAME solver COMMAND 007SolverTests "--gtest_output=json:${CMAKE_CURRENT_BINARY_DIR}/solver-test-results.json")
set_tests_properties(solver PROPERTIES TIMEOUT 60)

# Deliberately opt-in: neither the default build nor CTest runs the wide benchmark.
add_executable(007SolverBenchmark EXCLUDE_FROM_ALL tests/benchmark.cpp)
target_link_libraries(007SolverBenchmark PRIVATE solver_service_support GTest::gtest)
if(WIN32)
    target_link_libraries(007SolverBenchmark PRIVATE Psapi)
endif()
target_include_directories(007SolverBenchmark PRIVATE third_party/nlohmann_json/single_include)
target_compile_definitions(
    007SolverBenchmark
    PRIVATE
        TEST_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures/"
        BENCHMARK_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        BENCHMARK_CONFIGURATION="$<CONFIG>"
)
if(MSVC)
    target_compile_options(007SolverBenchmark PRIVATE /utf-8)
endif()
