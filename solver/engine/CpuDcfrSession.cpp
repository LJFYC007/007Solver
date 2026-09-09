#include "engine/CpuDcfrSession.h"
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
    , workerCount_(workers == 0 ? omp_get_max_threads() : workers)
{
    if (workerCount_ <= 0)
        throw std::invalid_argument("CPU DCFR worker count must be positive");
    regrets_.resize(traversal_.strategySize, 0.0f);
    strategySums_.resize(traversal_.strategySize, 0.0f);
    strategies_.resize(traversal_.strategySize);
    for (const HandTraversal::Node& node : traversal_.nodes)
    {
        if (node.kind == game::NodeKind::Decision)
        {
            std::fill_n(
                strategies_.data() + node.strategyOffset, node.childCount * traversal_.hands[node.actor].size(), 1.0f / node.childCount
            );
            for (const HandTraversal::Hand& hand : traversal_.hands[node.actor])
                if (!(hand.mask & node.boardMask))
                    ++infoSetCount_;
        }
    }
    for (std::size_t player = 0; player < 2; ++player)
        for (const auto& hand : traversal_.hands[player])
            divisors_[player].push_back(hand.opponentMass);
    for (std::size_t player = 0; player < 2; ++player)
        reach_[player].resize(traversal_.nodes.size() * traversal_.hands[player].size());
    values_.resize(traversal_.nodes.size() * std::max(traversal_.hands[0].size(), traversal_.hands[1].size()));
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
                reach_[player][hand] = player == updatingPlayer ? 1.0f : traversal_.hands[player][hand].weight;
        }

#pragma omp parallel num_threads(workerCount_)
        {
#pragma omp master
            workerCount_ = omp_get_num_threads();
            for (std::size_t depth = 0; depth < traversal_.levels.size(); ++depth)
            {
                const auto& level = traversal_.levels[depth];
#pragma omp for schedule(static)
                for (int index = 0; index < static_cast<int>(level.size()); ++index)
                    PropagateReach(level[index], updatingPlayer);
            }
            // Fold and showdown costs differ; workers take bounded terminal batches.
#pragma omp for schedule(dynamic, 64)
            for (int index = 0; index < static_cast<int>(traversal_.terminals.size()); ++index)
                EvaluateTerminal(traversal_.terminals[index], updatingPlayer);
            for (std::size_t depth = traversal_.levels.size(); depth > 0; --depth)
            {
                const auto& level = traversal_.levels[depth - 1];
#pragma omp for schedule(static)
                for (int index = 0; index < static_cast<int>(level.size()); ++index)
                    BackUp(level[index], updatingPlayer, positiveDiscount, averageDiscount);
            }
        }
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

void CpuDcfrSession::PropagateReach(std::uint32_t node, std::size_t updatingPlayer)
{
    for (std::size_t player = 0; player < 2; ++player)
        traversal_.PropagateReach(
            node,
            player,
            player != updatingPlayer,
            strategies_.data(),
            reach_[player].data() + node * traversal_.hands[player].size(),
            reach_[player].data(),
            player != updatingPlayer
        );
}

void CpuDcfrSession::EvaluateTerminal(std::uint32_t node, std::size_t updatingPlayer)
{
    traversal_.EvaluateTerminal(
        node,
        updatingPlayer,
        reach_[1 - updatingPlayer].data() + node * traversal_.hands[1 - updatingPlayer].size(),
        divisors_[updatingPlayer].data(),
        values_.data() + node * traversal_.hands[updatingPlayer].size()
    );
}

void CpuDcfrSession::BackUp(std::uint32_t nodeIndex, std::size_t updatingPlayer, float positiveDiscount, float averageDiscount)
{
    const HandTraversal::Node& node = traversal_.nodes[nodeIndex];
    const std::size_t count = traversal_.hands[updatingPlayer].size();
    float* values = values_.data() + nodeIndex * count;
    const bool updating = node.kind == game::NodeKind::Decision && node.actor == updatingPlayer;
    traversal_.BackUp(nodeIndex, updatingPlayer, strategies_.data(), false, values_.data());
    if (!updating)
        return;

    const double* ownReach = reach_[updatingPlayer].data() + nodeIndex * count;
    // Traverse contiguous hand rows while retaining each hand's action accumulation order.
    std::array<float, HandTraversal::kMaxHands> positiveRegrets;
    std::fill_n(positiveRegrets.data(), count, 0.0f);
    for (std::size_t action = 0; action < node.childCount; ++action)
    {
        const std::size_t offset = node.strategyOffset + action * count;
        const float* childValues = values_.data() + traversal_.children[node.childOffset + action] * count;
        for (std::size_t hand = 0; hand < count; ++hand)
        {
            const float regret = regrets_[offset + hand] + (childValues[hand] - values[hand]);
            regrets_[offset + hand] = regret * (regret > 0.0f ? positiveDiscount : 0.5f);
            positiveRegrets[hand] += std::max(0.0f, regrets_[offset + hand]);
            strategySums_[offset + hand] =
                static_cast<float>(averageDiscount * (strategySums_[offset + hand] + ownReach[hand] * strategies_[offset + hand]));
        }
    }
    for (std::size_t action = 0; action < node.childCount; ++action)
    {
        const std::size_t offset = node.strategyOffset + action * count;
        for (std::size_t hand = 0; hand < count; ++hand)
            strategies_[offset + hand] =
                positiveRegrets[hand] > 0.0f ? std::max(0.0f, regrets_[offset + hand]) / positiveRegrets[hand] : 1.0f / node.childCount;
    }
}

StrategySnapshot CpuDcfrSession::ExportStrategy() const
{
    std::vector<game::InfoSetKey> infoSets;
    std::vector<float> probabilities;
    infoSets.reserve(infoSetCount_);
    probabilities.reserve(strategySums_.size());
    std::vector<std::size_t> order(traversal_.nodes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return traversal_.nodes[a].id < traversal_.nodes[b].id; });
    for (const std::size_t index : order)
    {
        const HandTraversal::Node& node = traversal_.nodes[index];
        if (node.kind != game::NodeKind::Decision)
            continue;
        const auto& hands = traversal_.hands[node.actor];
        for (std::size_t hand = 0; hand < hands.size(); ++hand)
        {
            if (hands[hand].mask & node.boardMask)
                continue;
            double total = 0.0;
            for (std::size_t action = 0; action < node.childCount; ++action)
                total += strategySums_[node.strategyOffset + action * hands.size() + hand];
            infoSets.push_back({node.id, hands[hand].cards});
            for (std::size_t action = 0; action < node.childCount; ++action)
                probabilities.push_back(
                    total > 0.0 ? static_cast<float>(strategySums_[node.strategyOffset + action * hands.size() + hand] / total)
                                : 1.0f / node.childCount
                );
        }
    }
    return StrategySnapshot(problem_->game, infoSets, std::move(probabilities));
}
} // namespace solver::engine
