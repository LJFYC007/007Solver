#pragma once

#include "engine/HandTraversalData.h"
#include "engine/gpu/GpuTypes.h"
#include <algorithm>
#include <array>
#include <memory>
#include <vector>

namespace solver::engine::gpu
{
constexpr std::size_t kReadbackBytes = 16 * 1024 * 1024;
inline std::size_t TerminalSharedBytes(const State& state)
{
    // Group flags, opponent reach, forward/reverse rank prefixes (Fold stores one total
    // and 52 card masses instead), then each card's rank-ordered blocked reach prefix.
    return (kGroupFlags + state.stride + std::max(2 * state.stride, 53u) + 2 * state.stride + 52) * sizeof(float);
}
// One updating player's launch of a pass. Backup launches one lane per two of the
// updating player's hands. Only the opponent's reach is propagated, so a boundary Reach
// pass launches one lane per opponent hand and a deeper one launches the opponent's
// decisions, grouped by actor within the pass, with two hands per lane.
inline Pass LaunchPass(Pass pass, const State& shape, U32 player)
{
    if (pass.operation == Kernel::Backup)
        pass.lanes = (shape.hands[player] + 1) / 2;
    if (pass.operation != Kernel::Reach)
        return pass;
    const U32 opponent = 1 - player;
    if (!pass.boundary)
    {
        if (player == 0)
        {
            pass.offset += pass.split;
            pass.count -= pass.split;
        }
        else
            pass.count = pass.split;
    }
    pass.lanes = pass.boundary ? shape.hands[opponent] : (shape.hands[opponent] + 1) / 2;
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
    // One entry per scheduled work item, in pass order, so kernels index nodes by
    // pass offset, then one per decision a showdown's Terminal block backs up. A node
    // repeats at each of its passes; parents name a Backup entry or a trailing record.
    // Node stamps are indexed by the traversal node index, shared with the CPU layout.
    std::vector<Node> nodes;
    // Reach writes and Backup reads child slots without a dependent Node lookup.
    std::vector<U32> childSlots;
    std::vector<Hand> hands;
    // Each rank row stores ranks by hand, then every card's hands in rank order.
    std::vector<unsigned short> ranks;
    // Each rank row stores packed lower/upper bounds and per-card blocker positions
    // interleaved by hand, then the sorted hands, at an even pitch (State::orderPitch).
    std::vector<U32> order;
    std::vector<int> runouts;
    std::vector<U32> cards;
    std::vector<Pass> initialization;
    std::vector<Pass> passes;
    // Per pass, the earlier passes in other streams whose slot accesses conflict with it:
    // the graph edges a parallel executor needs beyond stream order.
    std::vector<std::vector<U32>> predecessors;
    State state{};
    std::size_t entries = 0; // 16-bit units of the regrets or sums buffer (HandTraversalData::strategySize)
    std::size_t slots = 0;
    std::size_t outcomeEntries = 0;
    std::array<BufferData, kBufferCount> Buffers() const;
    std::uint64_t DeviceBytes() const;
    std::uint64_t HostBytes() const;
};

// Both concrete implementations own their resident training state and scratch. Update
// may return before the device finishes; downloads, RootValues and Synchronize wait.
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
