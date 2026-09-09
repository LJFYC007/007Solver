file(
    GLOB SOLVER_LIB_SOURCES
    CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/analysis/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/engine/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/game/*.cpp"
)

add_library(solver_lib STATIC ${SOLVER_LIB_SOURCES})
target_compile_features(solver_lib PUBLIC cxx_std_17)
find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(solver_lib PRIVATE OpenMP::OpenMP_CXX)
if(MSVC)
    target_compile_options(solver_lib PRIVATE /utf-8)
endif()
target_include_directories(
    solver_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/SKPokerEval/src
)

file(
    GLOB SOLVER_SERVICE_SOURCES
    CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/service/*.cpp"
)

add_library(solver_service_support STATIC ${SOLVER_SERVICE_SOURCES})
target_compile_features(solver_service_support PUBLIC cxx_std_17)
target_link_libraries(solver_service_support PUBLIC solver_lib)
target_include_directories(
    solver_service_support
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/nlohmann_json/single_include
)
if(MSVC)
    target_compile_options(solver_service_support PRIVATE /utf-8)
endif()

add_executable(solver_service main.cpp)
target_link_libraries(solver_service PRIVATE solver_service_support)
if(MSVC)
    target_compile_options(solver_service PRIVATE /utf-8)
endif()

set(SOLVER_SERVICE_BUNDLE_DIR "${PROJECT_SOURCE_DIR}/desktop/src-tauri/binaries")
add_custom_target(
    stage_solver_service ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SOLVER_SERVICE_BUNDLE_DIR}"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:solver_service>"
        "${SOLVER_SERVICE_BUNDLE_DIR}/solver-service-x86_64-pc-windows-msvc.exe"
    DEPENDS solver_service
    VERBATIM
)
