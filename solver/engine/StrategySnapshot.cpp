#include "engine/StrategySnapshot.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
StrategySnapshot::StrategySnapshot(std::shared_ptr<const game::CompiledGame> game, std::vector<StrategyEntry> entries)
    : game_(std::move(game))
{
    if (!game_)
        throw std::invalid_argument("Strategy snapshot requires a compiled game");

    std::sort(
        entries.begin(),
        entries.end(),
        [](const StrategyEntry& left, const StrategyEntry& right)
        {
            if (left.infoSet.node != right.infoSet.node)
                return left.infoSet.node < right.infoSet.node;
            return left.infoSet.hand < right.infoSet.hand;
        }
    );

    std::size_t probabilityCount = 0;
    for (const StrategyEntry& entry : entries)
    {
        if (entry.probabilities.size() != game_->GetNode(entry.infoSet.node).BettingEdgeCount())
            throw std::invalid_argument("Strategy probability count does not match the node action count");
        probabilityCount += entry.probabilities.size();
    }
    probabilities_.reserve(probabilityCount);
    std::vector<game::InfoSetKey> infoSets;
    infoSets.reserve(entries.size());
    for (const StrategyEntry& entry : entries)
    {
        infoSets.push_back(entry.infoSet);
        probabilities_.insert(probabilities_.end(), entry.probabilities.begin(), entry.probabilities.end());
    }
    BuildIndex(infoSets);
}

StrategySnapshot::StrategySnapshot(
    std::shared_ptr<const game::CompiledGame> game,
    const std::vector<game::InfoSetKey>& infoSets,
    std::vector<float> probabilities
)
    : game_(std::move(game)), probabilities_(std::move(probabilities))
{
    BuildIndex(infoSets);
}

void StrategySnapshot::BuildIndex(const std::vector<game::InfoSetKey>& infoSets)
{
    if (!game_)
        throw std::invalid_argument("Strategy snapshot requires a compiled game");

    nodes_.resize(game_->NodeCount() + 1);
    hands_.reserve(infoSets.size());
    std::size_t nextNode = 0;
    std::size_t probabilityOffset = 0;
    const game::InfoSetKey* previous = nullptr;
    for (const game::InfoSetKey& infoSet : infoSets)
    {
        const game::GameNode& node = game_->GetNode(infoSet.node);
        if (node.Kind() != game::NodeKind::Decision || node.BettingEdgeCount() == 0)
            throw std::invalid_argument("Strategy entry must refer to a decision node with actions");
        if (core::Overlaps(infoSet.hand, node.State().board))
            throw std::invalid_argument("Strategy entry hand overlaps the public board");
        if (previous && previous->node == infoSet.node && previous->hand == infoSet.hand)
            throw std::invalid_argument("Strategy snapshot contains a duplicate infoset");
        if (previous && (infoSet.node < previous->node || (infoSet.node == previous->node && infoSet.hand < previous->hand)))
            throw std::invalid_argument("Packed strategy entries must be sorted by node and hand");

        const std::size_t actionCount = node.BettingEdgeCount();
        if (actionCount > probabilities_.size() - probabilityOffset)
            throw std::invalid_argument("Strategy probability count does not match the node action count");

        double totalProbability = 0.0;
        for (std::size_t actionIndex = 0; actionIndex < actionCount; ++actionIndex)
        {
            const float probability = probabilities_[probabilityOffset + actionIndex];
            if (!std::isfinite(probability) || probability < 0.0f)
                throw std::invalid_argument("Strategy probabilities must be finite and non-negative");
            totalProbability += probability;
        }
        constexpr double normalizationTolerance = 1e-5;
        if (std::abs(totalProbability - 1.0) > normalizationTolerance)
            throw std::invalid_argument("Strategy probabilities must sum to one");

        while (nextNode <= static_cast<std::size_t>(infoSet.node.Value()))
            nodes_[nextNode++] = {hands_.size(), probabilityOffset};
        hands_.push_back(infoSet.hand);
        probabilityOffset += actionCount;
        previous = &infoSet;
    }
    if (probabilityOffset != probabilities_.size())
        throw std::invalid_argument("Strategy probability count does not match the node action count");
    while (nextNode < nodes_.size())
        nodes_[nextNode++] = {hands_.size(), probabilityOffset};
}

const float* StrategySnapshot::FindStrategy(const game::InfoSetKey& infoSet) const
{
    const game::GameNode& node = game_->GetNode(infoSet.node);
    if (node.Kind() != game::NodeKind::Decision || node.BettingEdgeCount() == 0)
        throw std::invalid_argument("Strategy lookup requires a decision node with actions");

    const std::size_t nodeIndex = static_cast<std::size_t>(infoSet.node.Value());
    const NodeBlock& block = nodes_[nodeIndex];
    const auto begin = hands_.begin() + block.handOffset;
    const auto end = hands_.begin() + nodes_[nodeIndex + 1].handOffset;
    const auto hand = std::lower_bound(begin, end, infoSet.hand);
    if (hand == end || *hand != infoSet.hand)
        return nullptr;
    return probabilities_.data() + block.probabilityOffset + static_cast<std::size_t>(hand - begin) * node.BettingEdgeCount();
}

std::vector<float> StrategySnapshot::StrategyOrUniform(const game::InfoSetKey& infoSet) const
{
    const float* strategy = FindStrategy(infoSet);
    const std::size_t actionCount = game_->GetNode(infoSet.node).BettingEdgeCount();
    if (strategy)
        return std::vector<float>(strategy, strategy + actionCount);
    return std::vector<float>(actionCount, 1.0f / static_cast<float>(actionCount));
}
} // namespace solver::engine
