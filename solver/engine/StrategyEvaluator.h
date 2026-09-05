#pragma once

#include "engine/SolveProblem.h"
#include "engine/SolveResult.h"
#include "engine/StrategySnapshot.h"

namespace solver::engine
{
ExploitabilityMetrics EvaluateExploitability(const SolveProblem& problem, const StrategySnapshot& strategy);
} // namespace solver::engine
