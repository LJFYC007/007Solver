#pragma once

#include "engine/SolveProblem.h"
#include "engine/StrategySnapshot.h"
#include <array>
#include <map>
#include <memory>

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
// Borrows action-major cumulative strategies for the duration of this call.
ExploitabilityMetrics EvaluateAverageStrategy(const HandTraversal& traversal, const float* strategySums);

// Borrows one immutable result. A single EV worker reuses at most one board per
// street; traversal nodes, utility baselines and workspaces stay query-local.
class NodeStrategyEvaluator
{
public:
    NodeStrategyEvaluator(const SolveProblem& problem, const StrategySnapshot& strategy);
    // Fixed-policy net EV for board-compatible hands with positive opponent reach.
    // Caller decides eligibility from joint reach.
    std::map<core::HoleCards, float> Evaluate(game::NodeId node, core::PlayerId player);

private:
    const SolveProblem& problem_;
    const StrategySnapshot& strategy_;
    std::array<std::shared_ptr<const HandBoardData>, 3> boards_;
};

} // namespace solver::engine
