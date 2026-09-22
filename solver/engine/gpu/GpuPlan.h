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
    // Fold stores one total and 52 card masses, even for small ranges.
    return (state.stride + std::max(2 * state.stride, 53u)) * sizeof(float);
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
    std::vector<Node> nodes;
    // Reach writes and Backup reads child slots without a dependent Node lookup.
    std::vector<U32> childSlots;
    std::vector<Hand> hands;
    std::vector<unsigned short> ranks;
    // Each rank row stores sorted hands, then packed lower/upper bounds by hand.
    std::vector<U32> order;
    std::vector<int> runouts;
    std::vector<U32> cards;
    std::vector<U32> work;
    std::vector<Pass> initialization;
    std::vector<Pass> passes;
    State state{};
    std::size_t entries = 0;
    std::size_t slots = 0;
    std::size_t outcomeEntries = 0;
    std::array<BufferData, kBufferCount> Buffers() const;
    std::uint64_t DeviceBytes() const;
    std::uint64_t HostBytes() const;
};

// Both concrete implementations own their resident training state and scratch.
class Executor
{
public:
    virtual ~Executor() = default;
    virtual void Update(const State& state) = 0;
    virtual std::vector<float> RootValues(const State& state) = 0;
    virtual std::vector<float> DownloadSums(bool releaseTraining) = 0;
    virtual const char* Name() const = 0;
};
std::unique_ptr<Executor> MakeExecutor(const Plan& plan);
bool DeviceAvailable();
std::uint64_t DeviceMemoryBudget();
} // namespace solver::engine::gpu
