#pragma once

#include "engine/HandTraversalData.h"
#include "engine/gpu/GpuTypes.h"
#include <memory>
#include <vector>

namespace solver::engine::gpu
{
struct Plan
{
    explicit Plan(const HandTraversalData& data);
    std::vector<Node> nodes;
    std::vector<U32> edges;
    std::vector<Hand> hands;
    std::vector<unsigned short> ranks;
    // Each rank row stores sorted hands, then packed lower/upper bounds by hand.
    std::vector<U32> order;
    std::vector<int> runouts;
    std::vector<U32> cards;
    std::vector<U32> work;
    std::vector<Pass> passes;
    State state{};
    std::size_t entries = 0;
    std::size_t slots = 0;
    std::size_t outcomeEntries = 0;
    std::uint64_t DeviceBytes() const;
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
