#pragma once

#include "engine/HandTraversal.h"
#include "engine/StrategySnapshot.h"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace solver::engine
{
class CpuDcfrSession
{
public:
    // Zero workers selects the OpenMP runtime's default team size.
    explicit CpuDcfrSession(std::shared_ptr<const SolveProblem> problem, int workers = 0);

    // One iteration is a full update of one player, alternating across successive runs.
    void Run(int iterations, const std::function<void(int completedIterations)>& progressCallback = {});
    StrategySnapshot ExportStrategy() const;
    int CompletedIterations() const { return completedIterations_; }
    double TrainingTimeSeconds() const { return trainingTimeSeconds_; }
    int WorkerCount() const { return workerCount_; }

private:
    std::shared_ptr<const SolveProblem> problem_;
    const HandTraversal traversal_;
    std::array<std::vector<double>, 2> divisors_;
    // Action-major rows, with a fixed root-hand stride for the acting player.
    std::vector<float> regrets_;
    std::vector<float> strategySums_;
    std::vector<float> strategies_;
    std::array<std::vector<double>, 2> reach_;
    std::vector<float> values_;
    std::size_t infoSetCount_ = 0;
    int workerCount_;
    int completedIterations_ = 0;
    double trainingTimeSeconds_ = 0.0;

    void PropagateReach(std::uint32_t node, std::size_t updatingPlayer);
    void EvaluateTerminal(std::uint32_t node, std::size_t updatingPlayer);
    void BackUp(std::uint32_t node, std::size_t updatingPlayer, float positiveDiscount, float averageDiscount);
};
} // namespace solver::engine
