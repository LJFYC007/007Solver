// Compiles the shared kernels through their CUDA branch as host C++ and launches them
// with CudaExecutor::Dispatch's grid geometry.
#include "Dialect.h"
#include "FiberRuntime.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace emu_cuda
{
using namespace solver::engine::gpu;
struct Dim3
{
    unsigned x = 0, y = 0, z = 0;
};
Dim3 threadValue, blockValue, blockDimValue;
inline Dim3 ThreadIdx()
{
    auto* group = emu::Group::Active();
    return group ? Dim3{group->Local(), 0, 0} : threadValue;
}
// Dynamic shared memory; re-poisoned for every block.
constexpr std::size_t kSharedFloats = 1 << 16;
float groupMemory[kSharedFloats];

inline void __syncthreads()
{
    emu::Group::Current().Barrier();
}
inline int __syncthreads_or(int vote)
{
    return emu::Group::Current().BarrierOr(vote != 0) ? 1 : 0;
}
inline float __shfl_up_sync(unsigned mask, float value, unsigned delta)
{
    if (mask != 0xffffffffu)
        emu::Fail("partial shuffle mask");
    const float* lanes = emu::Group::Current().WarpCollect(value);
    const unsigned lane = emu::Group::Current().Local() % emu::kWarp;
    return lane >= delta ? lanes[lane - delta] : value;
}
inline float __shfl_sync(unsigned mask, float value, int source)
{
    if (mask != 0xffffffffu)
        emu::Fail("partial shuffle mask");
    return emu::Group::Current().WarpCollect(value)[unsigned(source) % emu::kWarp];
}
} // namespace emu_cuda

#define __device__
#define __forceinline__ inline
#define __global__
#define __shared__
#define threadIdx (ThreadIdx())
#define blockIdx blockValue
#define blockDim blockDimValue
namespace emu_cuda
{
#include "engine/gpu/GpuKernels.inc"
}
#undef threadIdx
#undef blockIdx
#undef blockDim

namespace emu
{
namespace
{
class CudaDialect final : public Dialect
{
public:
    const char* Name() const override { return "CUDA"; }
    void Dispatch(DeviceBuffers& b, const Pass& pass, U32 player, const State& shape, bool reverse) override
    {
        using namespace emu_cuda;
        if (!pass.count)
            return;
        // CudaExecutor::Dispatch geometry.
        const bool handTiles = pass.operation == Kernel::Backup;
        const auto handCount = shape.hands[player];
        const unsigned group = pass.operation == Kernel::Terminal ? 64u : handTiles ? (std::min(handCount, 256u) + 31) / 32 * 32 : 256u;
        const std::uint64_t threads = std::uint64_t(pass.count) * pass.lanes;
        const unsigned blocksX = handTiles                            ? pass.count
                                 : pass.operation == Kernel::Terminal ? pass.count
                                                                      : unsigned((threads + group - 1) / group);
        const unsigned blocksY = handTiles ? (handCount + group - 1) / group : 1u;
        if (group == 0 || group > 1024 || blocksY > 65535)
            Fail("invalid CUDA launch geometry");
        const std::size_t shared = pass.operation == Kernel::Terminal ? TerminalSharedBytes(shape) : 0;
        if (shared > sizeof(groupMemory))
            Fail("dynamic shared memory exceeds emulator capacity");
        if (shared > 99 * 1024)
            Fail("dynamic shared memory exceeds the 99 KiB opt-in limit of compute capability 8.9");
        blockDimValue = {group, 1, 1};
        const auto* statePointer = b.As<const State>(StateBuffer);
        const auto call = [&](auto kernel)
        {
            kernel(
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
                statePointer,
                pass
            );
        };
        const auto kernel = [&]
        {
            switch (pass.operation)
            {
            case Kernel::Reach:
                return &emu_cuda::Reach;
            case Kernel::Terminal:
                return &emu_cuda::Terminal;
            case Kernel::Backup:
                return &emu_cuda::Backup;
            default:
                return &emu_cuda::Outcomes;
            }
        }();
        const std::uint64_t blockCount = std::uint64_t(blocksX) * blocksY;
        for (std::uint64_t k = 0; k < blockCount; ++k)
        {
            const auto index = reverse ? blockCount - 1 - k : k;
            blockValue = {unsigned(index % blocksX), unsigned(index / blocksX), 0};
            if (pass.operation == Kernel::Terminal)
            {
                std::fill_n(groupMemory, shared / sizeof(float), std::numeric_limits<float>::quiet_NaN());
                group_.Run(group, reverse, [&](unsigned) { call(kernel); });
            }
            else
                for (unsigned t = 0; t < group; ++t)
                {
                    threadValue = {reverse ? group - 1 - t : t, 0, 0};
                    call(kernel);
                }
        }
    }

private:
    Group group_;
};
} // namespace
std::unique_ptr<Dialect> MakeCudaDialect()
{
    return std::make_unique<CudaDialect>();
}
} // namespace emu
