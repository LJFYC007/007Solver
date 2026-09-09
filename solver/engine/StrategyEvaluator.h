#pragma once

#include "engine/SolveProblem.h"
#include "engine/SolveResult.h"
#include "engine/StrategySnapshot.h"
#include <map>

namespace solver::engine
{
ExploitabilityMetrics EvaluateExploitability(const SolveProblem& problem, const StrategySnapshot& strategy);
// Fixed-policy net EV from the queried node, for all board-compatible hands with
// positive compatible opponent reach. Caller decides eligibility from joint reach.
std::map<core::HoleCards, float> EvaluateNodeStrategyEvs(
    const SolveProblem& problem,
    const StrategySnapshot& strategy,
    game::NodeId node,
    core::PlayerId player
);
} // namespace solver::engine
