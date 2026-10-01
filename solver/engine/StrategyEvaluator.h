#pragma once

#include "engine/SolveResult.h"
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

namespace solver::engine
{
struct ExploitabilityMetrics
{
    float player0BestResponseEv = 0.0f;
    float player1BestResponseEv = 0.0f;
    float exploitability = 0.0f;
};

struct HandTraversal;
struct HandBoardData;
ExploitabilityMetrics EvaluateExploitability(const SolveProblem& problem, const StrategySnapshot& strategy);
// Borrows quantized action-major cumulative strategies for the duration of this call. The traversal must start at the game root.
ExploitabilityMetrics EvaluateAverageStrategy(const HandTraversal& traversal, const std::uint16_t* strategySums);
// A player's game-root hand values under the best response, followed, when policy is set, by
// its values under the policy.
using RootValues = std::function<std::vector<float>(std::size_t player, bool policy)>;
// NashConv / 2, weighting game-root hand values by joint range mass. Zero-rake games skip
// policy values because the two policy utilities sum to zero.
ExploitabilityMetrics RootExploitability(const HandBoardData& tables, const RootValues& rootValues);

struct NodeStrategyValue
{
    float ev = 0.0f;
    // When the player acts at the node: the EV of taking each action and then following the policy.
    std::vector<float> actionEvs;
    // When the player acts at the node: the rake paid from it on in expectation, zero without rake.
    float expectedRake = 0.0f;
};

// Fixed-policy net EV at the traversal's root node for board-compatible hands with positive
// opponent reach; the caller decides eligibility from joint reach. The traversal must come from
// result's problem, rooted at the queried node. Workspaces stay call-local.
std::map<core::HoleCards, NodeStrategyValue> EvaluateNodeStrategy(
    const SolveResult& result,
    const HandTraversal& traversal,
    core::PlayerId player
);

} // namespace solver::engine
