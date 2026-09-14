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

MemoryEstimate EstimateCpuMemory(const SolveProblem& problem, int workers)
{
    if (!problem.game)
        throw std::invalid_argument("Memory estimate requires a compiled game");
    const auto size = problem.game->Size();
    const auto& board = problem.game->Spec().initialBoard;
    std::array<std::uint64_t, 2> hands{};
    for (std::uint8_t player = 0; player < 2; ++player)
        for (const auto& [hand, weight] : problem.ranges.For(core::PlayerId(player)).Entries())
            if (weight > 0.0f && !core::Overlaps(hand, board))
                ++hands[player];
    const std::uint64_t totalHands = hands[0] + hands[1];
    const std::uint64_t maxHands = std::max(hands[0], hands[1]);
    const std::uint64_t entries = size.actionEntries[0] * hands[0] + size.actionEntries[1] * hands[1];
    const std::uint64_t infosets = size.decisionNodes[0] * hands[0] + size.decisionNodes[1] * hands[1];
    const std::uint64_t decisions = size.decisionNodes[0] + size.decisionNodes[1];
    const int count = CpuWorkerCount(workers);
    const std::uint64_t layout = size.traversalNodes * (sizeof(HandTraversal::Node) + sizeof(std::uint32_t) + sizeof(std::uint64_t));
    const std::uint64_t ranks =
        1176 * (4 * totalHands + 2 * sizeof(std::vector<HandTraversal::RankedHand>)) + 2 * totalHands * sizeof(HandTraversal::Hand);
    const std::uint64_t stack = 8 * (size.depth + 1) * totalHands + size.depth * maxHands * (4 * size.maxActions + 8);
    // Include the bounded runout/regret scratch and per-team runtime overhead.
    const std::uint64_t workspace = stack * (count > 1 ? count + 1 : 1) + 52 * 4 * maxHands + (count + 1) * 128 * 1024;
    // During export, input infoset keys, snapshot hands and probability arrays coexist
    // with all three training tables. Sparse node indices allow vector growth headroom.
    const std::uint64_t exportPeak = 16 * entries + 12 * infosets + 80 * decisions + workspace;
    const std::uint64_t evaluationPeak = 8 * entries + 4 * infosets + 80 * decisions + stack;
    const std::uint64_t peak = size.storageBytes + layout + ranks + std::max(exportPeak, evaluationPeak);
    return {size.logicalNodes, size.topologyNodes, size.traversalNodes, entries, peak + peak / 8 + 64 * 1024 * 1024, count};
}
} // namespace solver::engine
