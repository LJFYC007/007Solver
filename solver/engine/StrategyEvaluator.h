#pragma once

#include "engine/SolveProblem.h"
#include "engine/StrategySnapshot.h"
#include <map>

namespace solver::engine
{
struct ExploitabilityMetrics
{
    float player0BestResponseEv = 0.0f;
    float player1BestResponseEv = 0.0f;
    float exploitability = 0.0f;
};

struct HandTraversal;
ExploitabilityMetrics EvaluateExploitability(const SolveProblem& problem, const StrategySnapshot& strategy);
// Borrows action-major cumulative strategies for the duration of this call.
ExploitabilityMetrics EvaluateAverageStrategy(const HandTraversal& traversal, const float* strategySums);
// Fixed-policy net EV from the queried node, for all board-compatible hands with
// positive compatible opponent reach. Caller decides eligibility from joint reach.
std::map<core::HoleCards, float> EvaluateNodeStrategyEvs(
    const SolveProblem& problem,
    const StrategySnapshot& strategy,
    game::NodeId node,
    core::PlayerId player
);
} // namespace solver::engine
