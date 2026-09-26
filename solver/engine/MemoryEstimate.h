#pragma once

#include "engine/SolveProblem.h"
#include <cstdint>

namespace solver::engine
{
struct MemoryEstimate
{
    std::uint64_t logicalNodes;
    std::uint64_t topologyNodes;
    std::uint64_t traversalNodes;
    std::uint64_t strategyEntries;
    std::uint64_t peakBytes;
    int workers;
};

// Counts available before either backend allocates a traversal or strategy state.
struct SolveSize
{
    game::GameTreeSize tree;
    std::array<std::size_t, 2> hands{};
    std::uint64_t strategyEntries = 0;
    std::uint64_t stateUnits = 0; // 16-bit units of the regrets or the strategy sums (HandTraversalData::StateUnits)
};
SolveSize MeasureSolveSize(const SolveProblem& problem);
// Resident regrets and strategy sums together.
inline std::uint64_t TrainingStateBytes(const SolveSize& size)
{
    return 2 * sizeof(std::uint16_t) * size.stateUnits;
}
// Adds headroom for allocator/runtime costs to a combined allocation peak.
MemoryEstimate MakeMemoryEstimate(const SolveSize& size, std::uint64_t peakBytes, int workers);

int CpuWorkerCount(int requested = 0);
// Conservative solve/export/evaluation peak, before allocating the active layout or
// strategy tables. Uses root-hand strides; includes headroom for allocator/runtime costs.
// User-driven node and equity queries after solving are not included.
MemoryEstimate EstimateCpuMemory(const SolveProblem& problem, int workers = 0);
} // namespace solver::engine
