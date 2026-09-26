#pragma once

#include "engine/HandTraversalData.h"
#include "engine/gpu/GpuTypes.h"
#include <algorithm>
#include <array>
#include <memory>
#include <vector>

namespace solver::engine::gpu
{
// Terminal shared memory in floats from the staged reach of the given opponent hands (a
// Terminal launch's lanes): that reach, its forward and reverse rank prefixes, each card's
// rank-ordered blocked reach after a leading zero, a fold's total and card masses, then a fold
// sibling's reach. The kernel's pointers and Plan::order follow it.
struct TerminalLayout
{
    explicit constexpr TerminalLayout(U32 hands)
        : forward(hands), reverse(2 * hands), runs(3 * hands), foldMasses(5 * hands + 52), end(foldMasses + 53 + hands)
    {}
    U32 forward, reverse, runs, foldMasses, end;
};
// One updating player's launch of a pass of Plan::passes: that player's items (Pass::begin
// and end) and the lanes each takes. Backup launches one lane per two of the updating
// player's hands. Terminal's lanes are the opponent's hands its shared memory holds. Only the
// opponent's reach is propagated, so Reach launches one lane per two opponent hands.
inline Pass LaunchPass(Pass pass, const State& shape, U32 player)
{
    pass.offset += pass.begin[player];
    pass.count = pass.end[player] - pass.begin[player];
    const U32 opponent = 1 - player;
    if (pass.operation == Kernel::Backup)
        pass.lanes = (shape.hands[player] + 1) / 2;
    else if (pass.operation == Kernel::Terminal)
        pass.lanes = shape.hands[opponent];
    else if (pass.operation == Kernel::Reach)
    {
        pass.lanes = (shape.hands[opponent] + 1) / 2;
        // Items at least a warp wide take whole warps, so no warp runs two items' different
        // paths; narrower ones stay packed rather than idle most of a warp.
        if (pass.lanes >= kWarpSize)
            pass.lanes = (pass.lanes + kWarpSize - 1) / kWarpSize * kWarpSize;
    }
    return pass;
}
// Each node's newest stamp is the larger of its entries in the stamps buffer's two halves.
inline std::vector<std::uint32_t> NewestStamps(const std::vector<std::uint32_t>& halves)
{
    const auto count = halves.size() / 2;
    std::vector<std::uint32_t> stamps(count);
    for (std::size_t i = 0; i < count; ++i)
        stamps[i] = std::max(halves[i], halves[count + i]);
    return stamps;
}

struct BufferData
{
    // Borrows a plan's upload source; null denotes device-only storage.
    const void* data = nullptr;
    std::size_t bytes = 0;
    std::size_t AllocationBytes() const { return std::max(bytes, sizeof(U32)); }
};

struct Plan
{
    explicit Plan(const HandTraversalData& data);
    // One entry per scheduled work item, in pass order, so kernels index nodes by pass
    // offset. A node repeats at each of its passes. Parents and node stamps are traversal
    // node indices, the stamps shared with the CPU layout.
    std::vector<Node> nodes;
    // Reach writes and Backup reads child slots without a dependent Node lookup.
    std::vector<U32> childSlots;
    std::vector<Hand> hands;
    // Each rank row stores ranks by hand, then every card's hands in rank order.
    std::vector<unsigned short> ranks;
    // Each rank row stores four entries per hand, the byte offsets from Terminal's staged reach
    // of what its showdown loads (see Terminal), then the sorted hands, at a pitch of a multiple
    // of four (State::orderPitch).
    std::vector<U32> order;
    std::vector<int> runouts;
    std::vector<U32> cards;
    std::vector<Pass> initialization;
    std::vector<Pass> passes;
    // Per pass, the earlier passes in other streams whose slot accesses conflict with it and
    // that neither stream order nor its other predecessors imply: the CUDA graph's edges beyond
    // stream order.
    std::vector<std::vector<U32>> predecessors;
    State state{};
    std::size_t entries = 0; // 16-bit units of the regrets or sums buffer (HandTraversalData::strategySize)
    std::size_t slots = 0;
    std::size_t outcomeEntries = 0;
    std::array<BufferData, kBufferCount> Buffers() const;
    std::uint64_t DeviceBytes() const;
    std::uint64_t HostBytes() const;
};

// The CUDA executor owns the resident training state and scratch; builds without CUDA link
// GpuUnavailable.cpp, which reports no device. Update may return before the device finishes;
// downloads, RootValues and Synchronize wait.
class Executor
{
public:
    virtual ~Executor() = default;
    virtual void Update(const State& state) = 0;
    virtual void Synchronize() = 0;
    virtual std::vector<float> RootValues(const State& state) = 0;
    virtual std::vector<std::uint16_t> DownloadSums(bool releaseTraining) = 0;
    virtual QuantizedState DownloadTraining() = 0;
    virtual void UploadTraining(const QuantizedState& state) = 0;
    virtual const char* Name() const = 0;
};
std::unique_ptr<Executor> MakeExecutor(const Plan& plan);
bool DeviceAvailable();
std::uint64_t DeviceMemoryBudget();
} // namespace solver::engine::gpu
