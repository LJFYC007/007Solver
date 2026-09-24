# Injected by tests/daily/run.py with -DCMAKE_PROJECT_007Solver_INCLUDE=<this file>, so the
# repository's own targets and CTest definitions build unchanged on a Linux host.
#   DAILY_BACKEND=cpu        GpuUnavailable.cpp, as the repository builds without a GPU toolkit
#   DAILY_BACKEND=emulation  GpuKernels.inc executed on the host through CUDA or Metal dialects
#   DAILY_BACKEND=cuda       CudaExecutor.cu built by nvcc (the WIN32 branch of solver/CMakeLists.txt)
set(DAILY_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(DAILY_BACKEND "cpu" CACHE STRING "cpu, emulation or cuda")

# The repository stages a Tauri sidecar only for macOS and Windows and stops configuration
# elsewhere. Swallow that single error; the sidecar is then staged without a target triple.
function(message mode)
    if(mode STREQUAL "FATAL_ERROR" AND "${ARGV1}" MATCHES "^Supported desktop targets")
        _message(STATUS "Daily harness: skipping desktop sidecar target check on this host")
        return()
    endif()
    _message(${ARGV})
endfunction()

# enable_language must run at directory scope, before the deferred patch.
if(DAILY_BACKEND STREQUAL "cuda")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

function(daily_patch)
    if(NOT DAILY_BACKEND STREQUAL "cpu")
        get_target_property(sources solver_lib SOURCES)
        list(FILTER sources EXCLUDE REGEX "GpuUnavailable\\.cpp$")
        set_target_properties(solver_lib PROPERTIES SOURCES "${sources}")
    endif()
    if(DAILY_BACKEND STREQUAL "emulation")
        # Metal kernels minus MSL [[attributes]], which host C++ cannot parse in macro arguments.
        file(READ "${PROJECT_SOURCE_DIR}/solver/engine/gpu/GpuKernels.inc" kernels)
        string(REGEX REPLACE "\\[\\[[^]]*\\]\\]" "" kernels "${kernels}")
        file(WRITE "${CMAKE_BINARY_DIR}/daily-generated/MetalKernels.inc" "${kernels}")
        set(emulation_sources "${DAILY_DIR}/gpu/EmulatedExecutor.cpp" "${DAILY_DIR}/gpu/CudaDialect.cpp" "${DAILY_DIR}/gpu/MetalDialect.cpp")
        target_sources(solver_lib PRIVATE ${emulation_sources})
        target_include_directories(solver_lib PRIVATE "${DAILY_DIR}/gpu" "${CMAKE_BINARY_DIR}/daily-generated")
        # The fiber switch is hand-written asm without shadow-stack support.
        set_source_files_properties(${emulation_sources} TARGET_DIRECTORY solver_lib PROPERTIES COMPILE_OPTIONS "-fcf-protection=none")
    elseif(DAILY_BACKEND STREQUAL "cuda")
        # Mirror the repository's CUDA settings from the tested revision instead of copying them.
        file(READ "${PROJECT_SOURCE_DIR}/solver/CMakeLists.txt" solver_cmake)
        if(NOT solver_cmake MATCHES "CUDA_ARCHITECTURES \"([^\"]+)\"")
            message(FATAL_ERROR "Daily harness: CUDA_ARCHITECTURES not found in solver/CMakeLists.txt; update tests/daily/cmake/daily.cmake")
        endif()
        set(architectures "${CMAKE_MATCH_1}")
        if(NOT solver_cmake MATCHES "\\$<\\$<COMPILE_LANGUAGE:CUDA>:([^>]+)>")
            message(FATAL_ERROR "Daily harness: CUDA compile options not found in solver/CMakeLists.txt; update tests/daily/cmake/daily.cmake")
        endif()
        set(cuda_options "${CMAKE_MATCH_1}")
        list(FILTER cuda_options EXCLUDE REGEX "^-Xcompiler=/") # MSVC-only host flags
        target_sources(solver_lib PRIVATE "${PROJECT_SOURCE_DIR}/solver/engine/gpu/CudaExecutor.cu")
        target_link_libraries(solver_lib PRIVATE CUDA::cudart_static)
        set_target_properties(
            solver_lib PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Static CUDA_ARCHITECTURES "${architectures}"
        )
        target_compile_options(solver_lib PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${cuda_options};-Xptxas=-v>")
        _message(STATUS "Daily harness CUDA: architectures ${architectures}; options ${cuda_options}")
    endif()
    # tests/benchmark.cpp reads process memory through Mach on non-Windows hosts.
    target_include_directories(007SolverBenchmark BEFORE PRIVATE "${DAILY_DIR}/shims")
    foreach(tool fingerprint parity planstats cpu_determinism)
        add_executable(daily_${tool} EXCLUDE_FROM_ALL "${DAILY_DIR}/tools/${tool}.cpp")
        target_link_libraries(daily_${tool} PRIVATE solver_service_support)
    endforeach()
    _message(STATUS "Daily harness backend: ${DAILY_BACKEND}")
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL daily_patch)
