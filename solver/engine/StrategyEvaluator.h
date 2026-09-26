#pragma once

#include "engine/SolveResult.h"
#include <array>
#include <cstdint>
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
// Weights each player's game-root best-response hand values by range weight and compatible opponent mass.
ExploitabilityMetrics RootExploitability(const HandBoardData& tables, const std::array<std::vector<float>, 2>& bestResponseValues);

// Fixed-policy net EV at node for board-compatible hands with positive opponent reach; the
// caller decides eligibility from joint reach. Board tables, traversal nodes and workspaces
// stay query-local.
std::map<core::HoleCards, float> EvaluateNodeStrategy(const SolveResult& result, game::NodeId node, core::PlayerId player);

} // namespace solver::engine
