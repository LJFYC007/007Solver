#pragma once
#include <cstdint>

namespace solver::service
{
// Leave one quarter of available physical memory for other applications.
std::uint64_t AvailableSolveMemory();
} // namespace solver::service
