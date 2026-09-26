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
    void Update(std::size_t player, const UpdateWeights& weights);
    ExploitabilityMetrics EvaluateExploitability() const;
    StrategySnapshot ExportStrategy() &&;
    QuantizedState ReadTrainingState() const { return state_; }
    void WriteTrainingState(const QuantizedState& state);
    const HandTraversal traversal_;
    std::array<std::vector<float>, 2> scales_;
    QuantizedState state_;
    HandTraversal::Workspace workspace_;
    std::vector<HandTraversal::Workspace> workers_;
    std::vector<float> rootValues_;
    int workerCount_;
};
} // namespace solver::engine
