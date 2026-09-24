#pragma once
#include "engine/gpu/GpuPlan.h"
#include <array>
#include <vector>

namespace emu
{
using namespace solver::engine::gpu;

// Host storage standing in for device buffers; exact sizes so sanitizers see overruns.
struct DeviceBuffers
{
    std::array<std::vector<unsigned char>, kBufferCount> data;
    template<typename T>
    T* As(std::size_t index)
    {
        return reinterpret_cast<T*>(data[index].data());
    }
};

// Launches one pass with a real backend's grid geometry and kernel source variant.
class Dialect
{
public:
    virtual ~Dialect() = default;
    virtual const char* Name() const = 0;
    virtual void Dispatch(DeviceBuffers& buffers, const Pass& pass, U32 player, const State& shape, bool reverse) = 0;
    virtual bool LaunchPassOnInitialization() const { return true; }
};
std::unique_ptr<Dialect> MakeCudaDialect();
std::unique_ptr<Dialect> MakeMetalDialect();
} // namespace emu
