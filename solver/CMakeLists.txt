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
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    # Homebrew's libomp is keg-only; let FindOpenMP see its headers and runtime.
    find_program(SOLVER_BREW_EXECUTABLE brew)
    if(SOLVER_BREW_EXECUTABLE)
        execute_process(
            COMMAND ${SOLVER_BREW_EXECUTABLE} --prefix libomp
            OUTPUT_VARIABLE SOLVER_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE SOLVER_LIBOMP_RESULT
        )
        if(SOLVER_LIBOMP_RESULT EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH "${SOLVER_LIBOMP_PREFIX}")
        endif()
    endif()
endif()
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

# Tauri sidecars are named for the native build target, not the executable suffix alone.
if(APPLE)
    set(SOLVER_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    if(CMAKE_OSX_ARCHITECTURES)
        set(SOLVER_ARCH "${CMAKE_OSX_ARCHITECTURES}")
    endif()
    if(SOLVER_ARCH STREQUAL "arm64")
        set(SOLVER_TARGET_TRIPLE "aarch64-apple-darwin")
    elseif(SOLVER_ARCH STREQUAL "x86_64")
        set(SOLVER_TARGET_TRIPLE "x86_64-apple-darwin")
    else()
        message(FATAL_ERROR "Build for one macOS architecture: arm64 or x86_64")
    endif()
elseif(WIN32 AND MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$")
    set(SOLVER_TARGET_TRIPLE "x86_64-pc-windows-msvc")
else()
    message(FATAL_ERROR "Supported desktop targets: macOS arm64/x86_64 and Windows x86_64 MSVC")
endif()

set(SOLVER_SERVICE_BUNDLE_DIR "${PROJECT_SOURCE_DIR}/desktop/src-tauri/binaries")
add_custom_target(
    stage_solver_service ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SOLVER_SERVICE_BUNDLE_DIR}"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:solver_service>"
        "${SOLVER_SERVICE_BUNDLE_DIR}/solver-service-${SOLVER_TARGET_TRIPLE}${CMAKE_EXECUTABLE_SUFFIX}"
    DEPENDS solver_service
    VERBATIM
)
