#include "engine/gpu/GpuPlan.h"
#include <stdexcept>

namespace solver::engine::gpu
{
bool DeviceAvailable()
{
    return false;
}
std::uint64_t DeviceMemoryBudget()
{
    return 0;
}
std::unique_ptr<Executor> MakeExecutor(const Plan&)
{
    throw std::runtime_error("This build has no GPU compute backend; build with CUDA or Metal, or select CPU");
}
} // namespace solver::engine::gpu
