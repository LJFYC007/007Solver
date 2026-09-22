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
    std::uint64_t infoSets = 0;
};
SolveSize MeasureSolveSize(const SolveProblem& problem);

int CpuWorkerCount(int requested = 0);
// Conservative solve/export/evaluation peak, before allocating the active layout or
// strategy tables. Uses root-hand strides; includes headroom for allocator/runtime costs.
// User-driven navigation and EV board caches after solving are not included.
MemoryEstimate EstimateCpuMemory(const SolveProblem& problem, int workers = 0);
} // namespace solver::engine
