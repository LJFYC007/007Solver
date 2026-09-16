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
    for (const HandTraversal::Node& node : traversal_.nodes)
    {
        if (node.kind == game::NodeKind::Decision)
        {
            for (const HandTraversal::Hand& hand : traversal_.hands[node.actor])
                if (!(hand.mask & node.boardMask))
                    ++infoSetCount_;
        }
    }
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
        const double t = completedIterations_ / 2 + 1.0;
        const double power = t * std::sqrt(t);
        const float positiveDiscount = static_cast<float>(power / (power + 1.0));
        const float averageDiscount = static_cast<float>((t / (t + 1.0)) * (t / (t + 1.0)));
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
    trainingTimeSeconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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
        std::vector<double>().swap(divisors);

    std::vector<StrategySnapshot::NodeBlock> nodes;
    std::vector<core::HoleCards> snapshotHands;
    snapshotHands.reserve(infoSetCount_);
    const auto maxHands = std::max(traversal_.hands[0].size(), traversal_.hands[1].size());
    std::vector<float> nodeSums(traversal_.maxActions * maxHands);
    std::size_t writeOffset = 0;
    // Copy each action-major node before writing its compact hand-major output.
    // Output never extends beyond the original node block, so later inputs survive.
    for (const HandTraversal::Node& node : traversal_.nodes)
    {
        if (node.kind != game::NodeKind::Decision)
            continue;
        const auto& hands = traversal_.hands[node.actor];
        std::copy_n(strategySums_.data() + node.strategyOffset, node.childCount * hands.size(), nodeSums.data());
        const auto handOffset = snapshotHands.size();
        const auto probabilityOffset = writeOffset;
        for (std::size_t hand = 0; hand < hands.size(); ++hand)
        {
            if (hands[hand].mask & node.boardMask)
                continue;
            double total = 0.0;
            for (std::size_t action = 0; action < node.childCount; ++action)
                total += nodeSums[action * hands.size() + hand];
            snapshotHands.push_back(hands[hand].cards);
            for (std::size_t action = 0; action < node.childCount; ++action)
                strategySums_[writeOffset++] =
                    total > 0.0 ? static_cast<float>(nodeSums[action * hands.size() + hand] / total) : 1.0f / node.childCount;
        }
        if (snapshotHands.size() != handOffset)
            nodes.push_back({node.id, handOffset, probabilityOffset, snapshotHands.size() - handOffset, node.childCount});
    }
    // Keep the allocation: shrinking capacity would require another large buffer.
    strategySums_.resize(writeOffset);
    return StrategySnapshot(problem_->game, std::move(nodes), std::move(snapshotHands), std::move(strategySums_));
}
} // namespace solver::engine
