#pragma once

#include "engine/SolveProblem.h"
#include "engine/StrategySnapshot.h"
#include <memory>
#include <string>

namespace solver::engine
{
struct ExploitabilityMetrics
{
    float player0BestResponseEv = 0.0f;
    float player1BestResponseEv = 0.0f;
    float exploitability = 0.0f;
};

struct SolveReport
{
    std::string algorithmId;
    std::string executionBackendId;
    int completedIterations = 0;
    double trainingTimeSeconds = 0.0;
    ExploitabilityMetrics metrics;
};

class SolveResult
{
public:
    SolveResult(std::shared_ptr<const SolveProblem> problem, StrategySnapshot strategy, SolveReport report);

    const SolveProblem& Problem() const { return *problem_; }
    const StrategySnapshot& Strategy() const { return strategy_; }
    const SolveReport& Report() const { return report_; }

private:
    std::shared_ptr<const SolveProblem> problem_;
    StrategySnapshot strategy_;
    SolveReport report_;
};
} // namespace solver::engine
