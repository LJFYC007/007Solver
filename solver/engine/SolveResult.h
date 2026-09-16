#pragma once

#include "engine/SolveProblem.h"
#include "engine/StrategySnapshot.h"
#include <memory>

namespace solver::engine
{
class SolveResult
{
public:
    SolveResult(std::shared_ptr<const SolveProblem> problem, StrategySnapshot strategy);

    const SolveProblem& Problem() const { return *problem_; }
    const StrategySnapshot& Strategy() const { return strategy_; }

private:
    std::shared_ptr<const SolveProblem> problem_;
    StrategySnapshot strategy_;
};
} // namespace solver::engine
