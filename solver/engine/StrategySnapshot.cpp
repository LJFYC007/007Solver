#include "engine/StrategySnapshot.h"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
StrategySnapshot::StrategySnapshot(std::shared_ptr<const game::CompiledGame> game, std::vector<StrategyEntry> entries)
    : game_(std::move(game)), strategiesByNode_(game_ ? game_->NodeCount() : 0)
{
    if (!game_)
        throw std::invalid_argument("Strategy snapshot requires a compiled game");

    for (StrategyEntry& entry : entries)
    {
        const game::GameNode& node = game_->GetNode(entry.infoSet.node);
        if (node.Kind() != game::NodeKind::Decision || node.BettingEdgeCount() == 0)
            throw std::invalid_argument("Strategy entry must refer to a decision node with actions");
        if (core::Overlaps(entry.infoSet.hand, node.State().board))
            throw std::invalid_argument("Strategy entry hand overlaps the public board");
        if (entry.probabilities.size() != node.BettingEdgeCount())
            throw std::invalid_argument("Strategy probability count does not match the node action count");

        double totalProbability = 0.0;
        for (const float probability : entry.probabilities)
        {
            if (!std::isfinite(probability) || probability < 0.0f)
                throw std::invalid_argument("Strategy probabilities must be finite and non-negative");
            totalProbability += probability;
        }
        constexpr double normalizationTolerance = 1e-5;
        if (std::abs(totalProbability - 1.0) > normalizationTolerance)
            throw std::invalid_argument("Strategy probabilities must sum to one");

        const bool inserted =
            strategiesByNode_[entry.infoSet.node.Value()].emplace(entry.infoSet.hand, std::move(entry.probabilities)).second;
        if (!inserted)
            throw std::invalid_argument("Strategy snapshot contains a duplicate infoset");
    }
}

const std::vector<float>* StrategySnapshot::FindStrategy(const game::InfoSetKey& infoSet) const
{
    const game::GameNode& node = game_->GetNode(infoSet.node);
    if (node.Kind() != game::NodeKind::Decision || node.BettingEdgeCount() == 0)
        throw std::invalid_argument("Strategy lookup requires a decision node with actions");

    const NodeStrategies& nodeStrategies = strategiesByNode_[infoSet.node.Value()];
    const auto strategy = nodeStrategies.find(infoSet.hand);
    return strategy == nodeStrategies.end() ? nullptr : &strategy->second;
}

std::vector<float> StrategySnapshot::StrategyOrUniform(const game::InfoSetKey& infoSet) const
{
    if (const std::vector<float>* strategy = FindStrategy(infoSet))
        return *strategy;

    const std::size_t actionCount = game_->GetNode(infoSet.node).BettingEdgeCount();
    return std::vector<float>(actionCount, 1.0f / static_cast<float>(actionCount));
}
} // namespace solver::engine
