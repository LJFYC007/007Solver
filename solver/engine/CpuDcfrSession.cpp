#include "engine/CpuDcfrSession.h"
#include "engine/MemoryEstimate.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <omp.h>

namespace solver::engine
{
CpuDcfrSession::CpuDcfrSession(std::shared_ptr<const SolveProblem> problem, int workers)
    : problem_(std::move(problem))
    , traversal_(problem_ ? *problem_ : throw std::invalid_argument("CPU DCFR session requires a solve problem"), game::NodeId(0))
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

        const HandTraversal::Update update =
            [&](std::uint32_t node, const double* ownReach, const float* strategy, const float* children, const float* values)
        { UpdateRegrets(node, updatingPlayer, ownReach, strategy, children, values, positiveDiscount, averageDiscount); };
        traversal_.Walk(
            0,
            updatingPlayer,
            nullptr,
            divisors_[updatingPlayer].data(),
            false,
            workspace_,
            0,
            rootValues_.data(),
            update,
            workerCount_ > 1 ? &workers_ : nullptr,
            regrets_.data()
        );
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

void CpuDcfrSession::UpdateRegrets(
    std::uint32_t nodeIndex,
    std::size_t updatingPlayer,
    const double* ownReach,
    const float* strategy,
    const float* children,
    const float* values,
    float positiveDiscount,
    float averageDiscount
)
{
    const HandTraversal::Node& node = traversal_.nodes[nodeIndex];
    const std::size_t count = traversal_.hands[updatingPlayer].size();
    // Traverse contiguous hand rows while retaining each hand's action accumulation order.
    for (std::size_t action = 0; action < node.childCount; ++action)
    {
        const std::size_t offset = node.strategyOffset + action * count;
        const float* childValues = children + action * count;
        for (std::size_t hand = 0; hand < count; ++hand)
        {
            const float regret = regrets_[offset + hand] + (childValues[hand] - values[hand]);
            regrets_[offset + hand] = regret * (regret > 0.0f ? positiveDiscount : 0.5f);
            strategySums_[offset + hand] =
                static_cast<float>(averageDiscount * (strategySums_[offset + hand] + ownReach[hand] * strategy[action * count + hand]));
        }
    }
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
