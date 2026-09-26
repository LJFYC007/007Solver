#include "engine/StrategySnapshot.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
std::uint64_t StrategySnapshot::EstimateStorageBytes(std::size_t nodes, std::size_t probabilities)
{
    // Node blocks grow incrementally during export; allow for vector capacity.
    return 2 * sizeof(NodeBlock) * nodes + sizeof(float) * probabilities;
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
    probabilities_.reset(new float[probabilityCount]);
    hands_.reserve(entries.size());
    std::optional<core::Board> board;
    std::size_t probabilityOffset = 0;
    for (const StrategyEntry& entry : entries)
    {
        if (nodes_.empty() || nodes_.back().node != entry.infoSet.node)
        {
            const auto node = game_->GetNode(entry.infoSet.node);
            if (node.Kind() != game::NodeKind::Decision || node.BettingEdgeCount() == 0)
                throw std::invalid_argument("Strategy entry must refer to a decision node with actions");
            nodes_.push_back({entry.infoSet.node, hands_.size(), probabilityOffset, 0, node.BettingEdgeCount()});
            board = node.State().board;
        }
        else if (hands_.back() == entry.infoSet.hand)
            throw std::invalid_argument("Strategy entry hands must be unique per node");
        if (core::Overlaps(entry.infoSet.hand, *board))
            throw std::invalid_argument("Strategy entry hand overlaps the public board");
        if (entry.probabilities.size() != nodes_.back().actionCount)
            throw std::invalid_argument("Strategy probability count does not match the node action count");
        float totalProbability = 0.0f;
        for (const float probability : entry.probabilities)
        {
            if (!std::isfinite(probability) || probability < 0.0f)
                throw std::invalid_argument("Strategy probabilities must be finite and non-negative");
            totalProbability += probability;
        }
        constexpr float normalizationTolerance = 1e-5f;
        if (std::abs(totalProbability - 1.0f) > normalizationTolerance)
            throw std::invalid_argument("Strategy probabilities must sum to one");
        ++nodes_.back().handCount;
        hands_.push_back(entry.infoSet.hand);
        std::copy(entry.probabilities.begin(), entry.probabilities.end(), probabilities_.get() + probabilityOffset);
        probabilityOffset += entry.probabilities.size();
    }
}

StrategySnapshot::StrategySnapshot(
    std::shared_ptr<const game::CompiledGame> game,
    std::vector<NodeBlock> nodes,
    std::vector<core::HoleCards> hands,
    std::unique_ptr<float[]> probabilities
)
    : game_(std::move(game)), nodes_(std::move(nodes)), hands_(std::move(hands)), probabilities_(std::move(probabilities))
{}

std::optional<StrategySnapshot::NodeStrategyView> StrategySnapshot::FindNodeStrategy(game::NodeId node) const
{
    const auto found =
        std::lower_bound(nodes_.begin(), nodes_.end(), node, [](const NodeBlock& block, game::NodeId id) { return block.node < id; });
    if (found == nodes_.end() || found->node != node)
        return std::nullopt;
    return NodeStrategyView{
        hands_.data() + found->handOffset, probabilities_.get() + found->probabilityOffset, found->handCount, found->actionCount
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
