// Compiles the shared kernels through their Metal branch as host C++ (MSL attributes are
// stripped at configure time) and launches them with MetalExecutor::Encode's geometry.
#include "Dialect.h"
#include "FiberRuntime.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace emu_metal
{
using namespace solver::engine::gpu;
using uint = unsigned int;
using ushort = unsigned short;
enum class mem_flags
{
    mem_none,
    mem_device,
    mem_threadgroup,
};
inline void threadgroup_barrier(mem_flags)
{
    emu::Group::Current().Barrier();
}
inline bool simd_any(bool value)
{
    const float* lanes = emu::Group::Current().WarpCollect(value ? 1.0f : 0.0f);
    for (unsigned i = 0; i < emu::kWarp; ++i)
        if (lanes[i] != 0.0f)
            return true;
    return false;
}
inline float simd_shuffle_up(float value, ushort delta)
{
    const float* lanes = emu::Group::Current().WarpCollect(value);
    const unsigned lane = emu::Group::Current().Local() % emu::kWarp;
    return lane >= delta ? lanes[lane - delta] : value;
}
inline float simd_shuffle(float value, ushort source)
{
    return emu::Group::Current().WarpCollect(value)[source % emu::kWarp];
}
inline float ldexp(float value, int exponent)
{
    return std::ldexp(value, exponent);
}
} // namespace emu_metal

#define __METAL_VERSION__ 230
#define device
#define kernel
#define constant const
#define threadgroup
namespace emu_metal
{
#include "MetalKernels.inc"
}
#undef device
#undef kernel
#undef constant
#undef threadgroup
#undef __METAL_VERSION__

namespace emu
{
namespace
{
class MetalDialect final : public Dialect
{
public:
    const char* Name() const override { return "Metal"; }
    void Dispatch(DeviceBuffers& b, const Pass& pass, U32, const State& shape, bool reverse) override
    {
        if (!pass.count)
            return;
        // MetalExecutor::Encode geometry on an Apple GPU: 32-wide SIMD groups, 1024-thread limit.
        const unsigned width = kWarp;
        const unsigned group = (pass.operation == Kernel::Terminal ? 128u : 256u) / width * width;
        const auto& state = *b.As<const State>(StateBuffer);
        const auto buffers = [&](auto... tail)
        {
            return std::make_tuple(
                b.As<const Node>(NodesBuffer),
                b.As<const U32>(ChildSlotsBuffer),
                b.As<const Hand>(HandsBuffer),
                b.As<const unsigned short>(RanksBuffer),
                b.As<const int>(RunoutsBuffer),
                b.As<const U32>(OrderBuffer),
                b.As<const U32>(CardsBuffer),
                b.As<U32>(OutcomesBuffer),
                b.As<float>(RegretsBuffer),
                b.As<float>(SumsBuffer),
                b.As<float>(ScratchBuffer),
                b.As<float>(ValuesBuffer),
                b.As<U32>(FlagsBuffer),
                b.As<U32>(StampsBuffer),
                tail...
            );
        };
        if (pass.operation == Kernel::Terminal)
        {
            const std::size_t floats = (TerminalSharedBytes(shape) + 15) / 16 * 16 / sizeof(float);
            if (floats * sizeof(float) > 32 * 1024)
                emu::Fail("threadgroup memory exceeds Metal's 32 KiB limit");
            for (U32 k = 0; k < pass.count; ++k)
            {
                const U32 groupIndex = reverse ? pass.count - 1 - k : k;
                // Exactly the threadgroup length the executor sets, poisoned so reads before writes show up.
                shared_.assign(floats, std::numeric_limits<float>::quiet_NaN());
                group_.Run(
                    group,
                    reverse,
                    [&](unsigned local) {
                        std::apply(
                            emu_metal::Terminal,
                            buffers(state, pass, groupIndex * group + local, width, shared_.data(), local, groupIndex, group)
                        );
                    }
                );
            }
            return;
        }
        // dispatchThreads launches exactly count * lanes threads (non-uniform threadgroups).
        const std::uint64_t threads = std::uint64_t(pass.count) * pass.lanes;
        for (std::uint64_t k = 0; k < threads; ++k)
        {
            const uint gid = uint(reverse ? threads - 1 - k : k);
            const auto arguments = buffers(state, pass, gid, width);
            switch (pass.operation)
            {
            case Kernel::Reach:
                std::apply(emu_metal::Reach, arguments);
                break;
            case Kernel::Backup:
                std::apply(emu_metal::Backup, arguments);
                break;
            default:
                std::apply(emu_metal::Outcomes, arguments);
                break;
            }
        }
    }

private:
    Group group_;
    std::vector<float> shared_;
};
} // namespace
std::unique_ptr<Dialect> MakeMetalDialect()
{
    return std::make_unique<MetalDialect>();
}
} // namespace emu
