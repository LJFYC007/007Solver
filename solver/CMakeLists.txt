file(
    GLOB SOLVER_LIB_SOURCES
    CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/analysis/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/engine/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/game/*.cpp"
)
list(APPEND SOLVER_LIB_SOURCES engine/gpu/GpuPlan.cpp)

option(SOLVER_ENABLE_GPU "Build the native CUDA or Metal DCFR backend when available" ON)
set(SOLVER_GPU_BACKEND "CPU")
if(SOLVER_ENABLE_GPU AND APPLE)
    enable_language(OBJCXX)
    set_source_files_properties(engine/gpu/MetalExecutor.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/engine/gpu/GpuTypes.h" GPU_TYPES)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/engine/gpu/GpuKernels.inc" GPU_KERNELS)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS engine/gpu/GpuTypes.h engine/gpu/GpuKernels.inc)
    configure_file(engine/gpu/MetalSource.h.in "${CMAKE_CURRENT_BINARY_DIR}/MetalSource.h" @ONLY)
    set(SOLVER_GPU_BACKEND "Metal")
elseif(SOLVER_ENABLE_GPU AND WIN32)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(SOLVER_GPU_BACKEND "CUDA")
    endif()
endif()
message(STATUS "DCFR backend: ${SOLVER_GPU_BACKEND} (recursive CPU always available)")
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    # Homebrew's libomp is keg-only; let FindOpenMP see its headers and link its
    # static runtime so packages do not depend on a Homebrew installation.
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
            set(OpenMP_libomp_LIBRARY "${SOLVER_LIBOMP_PREFIX}/lib/libomp.a" CACHE FILEPATH "OpenMP runtime" FORCE)
        endif()
    endif()
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)

file(
    GLOB SOLVER_SERVICE_SOURCES
    CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/service/*.cpp"
)

# Defines solver_lib, solver_service_support and solver_service with a name suffix.
# MSVC compiles solver_lib's C++ with the instruction set option `arch`.
function(add_solver_service suffix arch)
    add_library(solver_lib${suffix} STATIC ${SOLVER_LIB_SOURCES})
    target_compile_features(solver_lib${suffix} PUBLIC cxx_std_17)
    if(SOLVER_GPU_BACKEND STREQUAL "Metal")
        target_sources(solver_lib${suffix} PRIVATE engine/gpu/MetalExecutor.mm)
        set_target_properties(solver_lib${suffix} PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
        target_link_libraries(solver_lib${suffix} PRIVATE "-framework Metal" "-framework Foundation")
        target_include_directories(solver_lib${suffix} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    elseif(SOLVER_GPU_BACKEND STREQUAL "CUDA")
        target_sources(solver_lib${suffix} PRIVATE engine/gpu/CudaExecutor.cu)
        target_link_libraries(solver_lib${suffix} PRIVATE CUDA::cudart_static)
        # Native code for RTX 30, 40 and 50 series; other GPUs JIT the newest PTX they support.
        set_target_properties(
            solver_lib${suffix}
            PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Static CUDA_ARCHITECTURES "86;89-real;120"
        )
        target_compile_options(solver_lib${suffix} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
    else()
        target_sources(solver_lib${suffix} PRIVATE engine/gpu/GpuUnavailable.cpp)
    endif()
    target_link_libraries(solver_lib${suffix} PRIVATE OpenMP::OpenMP_CXX)
    if(MSVC)
        target_compile_options(solver_lib${suffix} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/utf-8;${arch}>")
    endif()
    target_include_directories(
        solver_lib${suffix}
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/third_party/SKPokerEval/src
    )

    add_library(solver_service_support${suffix} STATIC ${SOLVER_SERVICE_SOURCES})
    target_compile_features(solver_service_support${suffix} PUBLIC cxx_std_17)
    target_link_libraries(solver_service_support${suffix} PUBLIC solver_lib${suffix})
    target_include_directories(
        solver_service_support${suffix}
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/third_party/nlohmann_json/single_include
    )
    if(MSVC)
        target_compile_options(solver_service_support${suffix} PRIVATE /utf-8)
    endif()

    add_executable(solver_service${suffix} main.cpp)
    target_link_libraries(solver_service${suffix} PRIVATE solver_service_support${suffix})
    if(MSVC)
        target_compile_options(solver_service${suffix} PRIVATE /utf-8)
    endif()
endfunction()

add_solver_service("" /arch:AVX2)

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
set(SOLVER_STAGED_TARGETS solver_service)
set(SOLVER_STAGE_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SOLVER_SERVICE_BUNDLE_DIR}"
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:solver_service>"
        "${SOLVER_SERVICE_BUNDLE_DIR}/solver-service-${SOLVER_TARGET_TRIPLE}${CMAKE_EXECUTABLE_SUFFIX}"
)
if(MSVC)
    # The desktop runs this build on CPUs that lack the AVX2, FMA or BMI instructions
    # /arch:AVX2 code may use.
    add_solver_service(_sse2 /arch:SSE2)
    list(APPEND SOLVER_STAGED_TARGETS solver_service_sse2)
    # The installer carries the VC++ runtime DLLs that the services import.
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP ON)
    set(CMAKE_INSTALL_OPENMP_LIBRARIES ON)
    include(InstallRequiredSystemLibraries)
    list(FILTER CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS INCLUDE REGEX "/(msvcp140[^/]*|vcruntime140(_1)?|vcomp140)\\.dll$")
    if(NOT CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS MATCHES "/vcruntime140\\.dll" OR NOT CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS MATCHES "/vcomp140\\.dll")
        message(FATAL_ERROR "Missing VC++ runtime DLLs; found: ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}")
    endif()
    list(
        APPEND SOLVER_STAGE_COMMANDS
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:solver_service_sse2>"
            "${SOLVER_SERVICE_BUNDLE_DIR}/solver-service-sse2-${SOLVER_TARGET_TRIPLE}${CMAKE_EXECUTABLE_SUFFIX}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} "${SOLVER_SERVICE_BUNDLE_DIR}"
    )
endif()
add_custom_target(stage_solver_service ALL ${SOLVER_STAGE_COMMANDS} DEPENDS ${SOLVER_STAGED_TARGETS} VERBATIM)
