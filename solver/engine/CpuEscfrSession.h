#pragma once

#include "core/Card.h"
#include "core/PokerTypes.h"
#include "engine/SolveProblem.h"
#include "engine/StrategySnapshot.h"
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

namespace solver::engine
{
using ProgressCallback = std::function<void(int completedIterations)>;

class CpuEscfrSession
{
public:
    explicit CpuEscfrSession(std::shared_ptr<const SolveProblem> problem);

    // Each iteration updates one player; successive runs continue the same training state.
    void Run(int iterations, const ProgressCallback& progressCallback = {});
    StrategySnapshot ExportStrategy() const;
    int CompletedIterations() const { return completedIterations_; }
    double TrainingTimeSeconds() const { return trainingTimeSeconds_; }

private:
    struct InfoSetState
    {
        std::vector<float> regrets;
        std::vector<float> strategySum;
    };

    std::shared_ptr<const SolveProblem> problem_;
    std::default_random_engine rng_{42};
    std::vector<std::unordered_map<core::HoleCards, InfoSetState, core::HoleCardsHash>> infoSetsByNode_;
    int completedIterations_ = 0;
    double trainingTimeSeconds_ = 0.0;

    float SampleTraverse(game::NodeId node, core::PlayerId updatingPlayer, core::HoleCards player0Hand, core::HoleCards player1Hand);
    InfoSetState& GetInfoSet(game::NodeId node, core::HoleCards hand);
};
} // namespace solver::engine
