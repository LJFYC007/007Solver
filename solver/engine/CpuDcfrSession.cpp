#include "engine/CpuDcfrSession.h"
#include "engine/MemoryEstimate.h"
#include "engine/StrategyEvaluator.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
CpuDcfrSession::CpuDcfrSession(const SolveProblem& problem, int workers)
    : traversal_(problem, problem.game->Root(), true), workerCount_(CpuWorkerCount(workers))
{
    if (workerCount_ <= 0)
        throw std::invalid_argument("CPU DCFR worker count must be positive");
    state_.regrets.resize(traversal_.strategySize, 0);
    state_.strategySums.resize(traversal_.strategySize, 0);
    state_.stamps.resize(traversal_.nodes.size(), 0);
    for (std::size_t player = 0; player < 2; ++player)
        for (const auto& hand : traversal_.hands[player])
            divisors_[player].push_back(hand.opponentMass);
    workspace_ = traversal_.MakeWorkspace(workerCount_ > 1);
    if (workerCount_ > 1)
        for (int worker = 0; worker < workerCount_; ++worker)
            workers_.push_back(traversal_.MakeWorkspace());
    rootValues_.resize(std::max(traversal_.hands[0].size(), traversal_.hands[1].size()));
}

void CpuDcfrSession::Update(std::size_t updatingPlayer, const UpdateWeights& weights)
{
    const auto& opponentHands = traversal_.hands[1 - updatingPlayer];
    for (std::size_t hand = 0; hand < opponentHands.size(); ++hand)
        workspace_.reach[hand] = opponentHands[hand].weight;

    HandTraversal::TrainState train{state_.regrets.data(), state_.strategySums.data(), state_.stamps.data(), weights};
    traversal_.WalkTraining(updatingPlayer, divisors_[updatingPlayer].data(), workspace_, rootValues_.data(), workers_, train);
}

ExploitabilityMetrics CpuDcfrSession::EvaluateExploitability() const
{
    return EvaluateAverageStrategy(traversal_, state_.strategySums.data());
}

void CpuDcfrSession::WriteTrainingState(const QuantizedState& state)
{
    if (state.regrets.size() != traversal_.strategySize || state.strategySums.size() != traversal_.strategySize ||
        state.stamps.size() != traversal_.nodes.size())
        throw std::invalid_argument("Training state does not match the CPU strategy layout");
    state_ = state;
}

StrategySnapshot CpuDcfrSession::ExportStrategy() &&
{
    // Only the cumulative strategy and traversal layout are needed below. Release
    // training allocations before the snapshot probability and index arrays coexist.
    std::vector<std::int16_t>().swap(state_.regrets);
    std::vector<std::uint32_t>().swap(state_.stamps);
    workspace_ = {};
    std::vector<HandTraversal::Workspace>().swap(workers_);
    std::vector<float>().swap(rootValues_);
    for (auto& divisors : divisors_)
        std::vector<float>().swap(divisors);

    return traversal_.Data().ExportStrategy(std::move(state_.strategySums));
}
} // namespace solver::engine
