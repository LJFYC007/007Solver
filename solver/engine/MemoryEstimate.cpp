#include "engine/MemoryEstimate.h"
#include "engine/HandTraversal.h"
#include <algorithm>
#include <stdexcept>
#include <omp.h>

namespace solver::engine
{
int CpuWorkerCount(int requested)
{
    if (requested < 0)
        throw std::invalid_argument("CPU worker count cannot be negative");
    return std::min(49, requested == 0 ? omp_get_max_threads() : requested);
}

SolveSize MeasureSolveSize(const SolveProblem& problem)
{
    if (!problem.game)
        throw std::invalid_argument("Memory estimate requires a compiled game");
    SolveSize size{problem.game->Size()};
    const auto& board = problem.game->Spec().initialBoard;
    for (std::uint8_t player = 0; player < 2; ++player)
    {
        for (const auto& [hand, weight] : problem.ranges.For(core::PlayerId(player)).Entries())
            if (weight > 0.0f && !core::Overlaps(hand, board))
                ++size.hands[player];
        size.strategyEntries += size.tree.actionEntries[player] * size.hands[player];
        size.infoSets += size.tree.decisionNodes[player] * size.hands[player];
    }
    return size;
}

MemoryEstimate MakeMemoryEstimate(const SolveSize& size, std::uint64_t peakBytes, int workers)
{
    const auto& tree = size.tree;
    return {
        tree.logicalNodes,
        tree.topologyNodes,
        tree.traversalNodes,
        size.strategyEntries,
        peakBytes + peakBytes / 8 + 64 * 1024 * 1024,
        workers
    };
}

MemoryEstimate EstimateCpuMemory(const SolveProblem& problem, int workers)
{
    const auto counts = MeasureSolveSize(problem);
    const auto& size = counts.tree;
    const auto& hands = counts.hands;
    const std::uint64_t maxHands = std::max(hands[0], hands[1]);
    const std::uint64_t entries = counts.strategyEntries;
    const std::uint64_t infosets = counts.infoSets;
    const std::uint64_t decisions = size.decisionNodes[0] + size.decisionNodes[1];
    const int count = CpuWorkerCount(workers);
    const auto storage = HandTraversal::EstimateStorage(*problem.game, hands, true);
    const auto rootVectors = sizeof(float) * (hands[0] + hands[1] + maxHands);
    // Include the bounded runout/regret scratch and per-team runtime overhead.
    const std::uint64_t workspace = storage.workspaceBytes * (count > 1 ? count + 1 : 1) + (count > 1 ? storage.parallelValuesBytes : 0) +
                                    (count + 1) * 128 * 1024 + rootVectors;
    // Training retains regrets, strategy sums and node stamps; current policies live in depth rows.
    const std::uint64_t trainingPeak =
        2 * sizeof(float) * entries + sizeof(std::uint32_t) * size.traversalNodes + workspace + storage.flopOutcomesBytes;
    // Checkpoints borrow sums while all training allocations remain resident.
    const auto checkpointPeak = trainingPeak + storage.workspaceBytes + rootVectors;
    // Final export releases regrets and workspaces before allocating the snapshot.
    // Snapshot probabilities reuse the strategy-sum allocation; indices are built directly.
    const std::uint64_t snapshot = StrategySnapshot::EstimateStorageBytes(decisions, infosets, entries);
    const std::uint64_t exportPeak = snapshot + sizeof(float) * size.maxActions * maxHands + storage.flopOutcomesBytes;
    const std::uint64_t evaluationPeak = snapshot + storage.workspaceBytes + rootVectors;
    const std::uint64_t peak = size.storageBytes + storage.fixedBytes + std::max({checkpointPeak, exportPeak, evaluationPeak});
    return MakeMemoryEstimate(counts, peak, count);
}
} // namespace solver::engine
