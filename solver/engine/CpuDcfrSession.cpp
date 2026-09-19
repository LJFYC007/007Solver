#include "engine/CpuDcfrSession.h"
#include "engine/MemoryEstimate.h"
#include "engine/StrategyEvaluator.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
CpuDcfrSession::CpuDcfrSession(std::shared_ptr<const SolveProblem> problem, int workers)
    : problem_(std::move(problem))
    , traversal_(problem_ ? *problem_ : throw std::invalid_argument("CPU DCFR session requires a solve problem"), game::NodeId(0), true)
    , workerCount_(CpuWorkerCount(workers))
{
    if (workerCount_ <= 0)
        throw std::invalid_argument("CPU DCFR worker count must be positive");
    regrets_.resize(traversal_.strategySize, 0.0f);
    strategySums_.resize(traversal_.strategySize, 0.0f);
    for (std::size_t player = 0; player < 2; ++player)
        for (const auto& hand : traversal_.hands[player])
            divisors_[player].push_back(hand.opponentMass);
    workspace_ = traversal_.MakeWorkspace(workerCount_ > 1);
    if (workerCount_ > 1)
        for (int worker = 0; worker < workerCount_; ++worker)
            workers_.push_back(traversal_.MakeWorkspace());
    rootValues_.resize(std::max(traversal_.hands[0].size(), traversal_.hands[1].size()));
}

void CpuDcfrSession::Run(int iterations, const std::function<void(int)>& progressCallback)
{
    if (iterations <= 0 || iterations > std::numeric_limits<int>::max() - completedIterations_)
        throw std::invalid_argument("DCFR iterations must be positive and fit the completed iteration counter");
    const auto start = std::chrono::steady_clock::now();
    auto lastProgress = start;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const std::size_t updatingPlayer = static_cast<std::size_t>(completedIterations_ % 2);
        const float t = completedIterations_ / 2 + 1.0f;
        const float power = t * std::sqrt(t);
        const float positiveDiscount = power / (power + 1.0f);
        const float averageDiscount = (t / (t + 1.0f)) * (t / (t + 1.0f));
        for (std::size_t player = 0; player < 2; ++player)
        {
            for (std::size_t hand = 0; hand < traversal_.hands[player].size(); ++hand)
                workspace_.reach[player][hand] = player == updatingPlayer ? 1.0f : traversal_.hands[player][hand].weight;
        }

        HandTraversal::TrainState train{regrets_.data(), strategySums_.data(), positiveDiscount, averageDiscount};
        traversal_.WalkTraining(updatingPlayer, divisors_[updatingPlayer].data(), workspace_, rootValues_.data(), workers_, train);
        ++completedIterations_;
        if (progressCallback)
        {
            const auto now = std::chrono::steady_clock::now();
            if (iteration + 1 == iterations || now - lastProgress >= std::chrono::milliseconds(250))
            {
                progressCallback(completedIterations_);
                lastProgress = now;
            }
        }
    }
    trainingTimeSeconds_ += std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
}

ExploitabilityMetrics CpuDcfrSession::EvaluateExploitability() const
{
    return EvaluateAverageStrategy(traversal_, strategySums_.data());
}

StrategySnapshot CpuDcfrSession::ExportStrategy() &&
{
    // Only the cumulative strategy and traversal layout are needed below. Release
    // training allocations before the snapshot probability and index arrays coexist.
    std::vector<float>().swap(regrets_);
    workspace_ = {};
    std::vector<HandTraversal::Workspace>().swap(workers_);
    std::vector<float>().swap(rootValues_);
    for (auto& divisors : divisors_)
        std::vector<float>().swap(divisors);

    return traversal_.Data().ExportStrategy(std::move(strategySums_));
}
} // namespace solver::engine
