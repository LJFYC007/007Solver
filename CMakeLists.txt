cmake_minimum_required(VERSION 3.20)
project(007Solver LANGUAGES CXX)

if(MSVC)
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
add_test(NAME solver COMMAND 007SolverTests)
set_tests_properties(solver PROPERTIES TIMEOUT 60)
