#include "engine/StrategySnapshot.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
std::uint64_t StrategySnapshot::EstimateStorageBytes(std::size_t nodes, std::size_t hands, std::size_t probabilities)
{
    // Node blocks grow incrementally during export; allow for vector capacity.
    return 2 * sizeof(NodeBlock) * nodes + sizeof(core::HoleCards) * hands + sizeof(float) * probabilities;
}

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
        probabilityCount += entry.probabilities.size();
    probabilities_.reserve(probabilityCount);
    hands_.reserve(entries.size());
    for (const StrategyEntry& entry : entries)
    {
        if (nodes_.empty() || nodes_.back().node != entry.infoSet.node)
            nodes_.push_back(
                {entry.infoSet.node, hands_.size(), probabilities_.size(), 0, game_->GetNode(entry.infoSet.node).BettingEdgeCount()}
            );
        if (entry.probabilities.size() != nodes_.back().actionCount)
            throw std::invalid_argument("Strategy probability count does not match the node action count");
        ++nodes_.back().handCount;
        hands_.push_back(entry.infoSet.hand);
        probabilities_.insert(probabilities_.end(), entry.probabilities.begin(), entry.probabilities.end());
    }
    Validate();
}

StrategySnapshot::StrategySnapshot(
    std::shared_ptr<const game::CompiledGame> game,
    std::vector<NodeBlock> nodes,
    std::vector<core::HoleCards> hands,
    std::vector<float> probabilities
)
    : game_(std::move(game)), nodes_(std::move(nodes)), hands_(std::move(hands)), probabilities_(std::move(probabilities))
{
    Validate();
}

void StrategySnapshot::Validate() const
{
    if (!game_)
        throw std::invalid_argument("Strategy snapshot requires a compiled game");

    std::size_t handOffset = 0, probabilityOffset = 0;
    const NodeBlock* previous = nullptr;
    for (const NodeBlock& block : nodes_)
    {
        const auto node = game_->GetNode(block.node);
        if (node.Kind() != game::NodeKind::Decision || node.BettingEdgeCount() == 0)
            throw std::invalid_argument("Strategy entry must refer to a decision node with actions");
        if (previous && !(previous->node < block.node))
            throw std::invalid_argument("Packed strategy nodes must be sorted and unique");
        if (block.handOffset != handOffset || block.probabilityOffset != probabilityOffset ||
            block.handCount > hands_.size() - handOffset || block.actionCount != node.BettingEdgeCount() ||
            block.handCount > (probabilities_.size() - probabilityOffset) / block.actionCount)
            throw std::invalid_argument("Strategy block size does not match its hands and actions");
        for (std::size_t hand = 0; hand < block.handCount; ++hand)
        {
            const auto cards = hands_[handOffset + hand];
            if (core::Overlaps(cards, node.State().board))
                throw std::invalid_argument("Strategy entry hand overlaps the public board");
            if (hand > 0 && !(hands_[handOffset + hand - 1] < cards))
                throw std::invalid_argument("Packed strategy hands must be sorted and unique");
            float totalProbability = 0.0f;
            for (std::size_t action = 0; action < block.actionCount; ++action)
            {
                const float probability = probabilities_[probabilityOffset++];
                if (!std::isfinite(probability) || probability < 0.0f)
                    throw std::invalid_argument("Strategy probabilities must be finite and non-negative");
                totalProbability += probability;
            }
            constexpr float normalizationTolerance = 1e-5f;
            if (std::abs(totalProbability - 1.0f) > normalizationTolerance)
                throw std::invalid_argument("Strategy probabilities must sum to one");
        }
        handOffset += block.handCount;
        previous = &block;
    }
    if (handOffset != hands_.size() || probabilityOffset != probabilities_.size())
        throw std::invalid_argument("Strategy probability count does not match the node action count");
}

std::optional<StrategySnapshot::NodeStrategyView> StrategySnapshot::FindNodeStrategy(game::NodeId node) const
{
    const auto found =
        std::lower_bound(nodes_.begin(), nodes_.end(), node, [](const NodeBlock& block, game::NodeId id) { return block.node < id; });
    if (found == nodes_.end() || found->node != node)
        return std::nullopt;
    return NodeStrategyView{
        hands_.data() + found->handOffset, probabilities_.data() + found->probabilityOffset, found->handCount, found->actionCount
    };
}

const float* StrategySnapshot::FindStrategy(const game::InfoSetKey& infoSet) const
{
    const auto block = FindNodeStrategy(infoSet.node);
    if (!block)
    {
        if (game_->GetNode(infoSet.node).Kind() != game::NodeKind::Decision)
            throw std::invalid_argument("Strategy lookup requires a decision node with actions");
        return nullptr;
    }
    const auto end = block->hands + block->handCount;
    const auto hand = std::lower_bound(block->hands, end, infoSet.hand);
    if (hand == end || *hand != infoSet.hand)
        return nullptr;
    return block->probabilities + static_cast<std::size_t>(hand - block->hands) * block->actionCount;
}

float StrategySnapshot::ActionProbability(const game::InfoSetKey& infoSet, std::size_t action) const
{
    const float* strategy = FindStrategy(infoSet);
    return strategy ? strategy[action] : 1.0f / (game_->GetNode(infoSet.node).BettingEdgeCount());
}

std::vector<float> StrategySnapshot::StrategyOrUniform(const game::InfoSetKey& infoSet) const
{
    const float* strategy = FindStrategy(infoSet);
    const std::size_t actionCount = game_->GetNode(infoSet.node).BettingEdgeCount();
    if (strategy)
        return std::vector<float>(strategy, strategy + actionCount);
    return std::vector<float>(actionCount, 1.0f / actionCount);
}
} // namespace solver::engine
