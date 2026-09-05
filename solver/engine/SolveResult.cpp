#include "engine/SolveResult.h"
#include <stdexcept>
#include <utility>

namespace solver::engine
{
SolveResult::SolveResult(std::shared_ptr<const SolveProblem> problem, StrategySnapshot strategy, SolveReport report)
    : problem_(std::move(problem)), strategy_(std::move(strategy)), report_(std::move(report))
{
    if (!problem_ || !problem_->game)
        throw std::invalid_argument("Solve result requires a solve problem");
    if (problem_->game.get() != &strategy_.Game())
        throw std::invalid_argument("Solve result strategy belongs to a different compiled game");
}
} // namespace solver::engine
