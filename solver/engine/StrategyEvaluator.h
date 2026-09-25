#pragma once

#include "engine/SolveResult.h"
#include <array>
#include <cstdint>
#include <map>
#include <memory>
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

// Borrows one immutable result. A single EV worker reuses at most one board per
// street; traversal nodes, utility baselines and workspaces stay query-local.
class NodeStrategyEvaluator
{
public:
    explicit NodeStrategyEvaluator(const SolveResult& result);
    // Fixed-policy net EV for board-compatible hands with positive opponent reach.
    // Caller decides eligibility from joint reach.
    std::map<core::HoleCards, float> Evaluate(game::NodeId node, core::PlayerId player);

private:
    const SolveProblem& problem_;
    const StrategySnapshot& strategy_;
    std::array<std::shared_ptr<const HandBoardData>, 3> boards_;
};

} // namespace solver::engine
