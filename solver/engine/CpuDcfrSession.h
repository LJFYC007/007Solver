#pragma once

#include "engine/HandTraversal.h"
#include "engine/StrategyEvaluator.h"
#include "engine/StrategySnapshot.h"
#include <array>
#include <cstdint>
#include <vector>

namespace solver::engine
{
class CpuDcfrSession
{
public:
    // Zero workers selects the OpenMP runtime's default team size.
    explicit CpuDcfrSession(const SolveProblem& problem, int workers = 0);

    // Configured workspace/team limit; the OpenMP runtime may use fewer threads.
    int WorkerCount() const { return workerCount_; }

private:
    friend class DcfrSession;
    void Update(std::size_t player, float positiveDiscount, float averageDiscount);
    ExploitabilityMetrics EvaluateExploitability() const;
    StrategySnapshot ExportStrategy() &&;
    TrainingState ReadTrainingState() const { return {regrets_, strategySums_}; }
    void WriteTrainingState(const TrainingState& state);
    const HandTraversal traversal_;
    std::array<std::vector<float>, 2> divisors_;
    // Action-major rows, with a fixed root-hand stride for the acting player.
    std::vector<float> regrets_;
    std::vector<float> strategySums_;
    HandTraversal::Workspace workspace_;
    std::vector<HandTraversal::Workspace> workers_;
    std::vector<float> rootValues_;
    int workerCount_;
};
} // namespace solver::engine
