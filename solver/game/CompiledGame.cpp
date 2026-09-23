#include "game/CompiledGame.h"
#include "core/Evaluator.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace solver::game
{
namespace
{
constexpr std::size_t kHandCount = 52 * 51 / 2;
constexpr std::size_t kRunoutCount = 49 * 48 / 2;
// SKPokerEval ranks are at most 7462; blocked slots must never compare as real ranks.
constexpr std::uint16_t kBlockedRank = std::numeric_limits<std::uint16_t>::max();
} // namespace

GameNode::GameNode(
    const std::vector<BettingTopologyNode>* topology,
    std::uint32_t shape,
    NodeId id,
    core::Board board,
    std::optional<ParentEdge> parent
)
    : topology_(topology), shape_(shape), id_(id), state_((*topology)[shape].state), parent_(parent)
{
    state_.board = board;
}

const TerminalOutcome& GameNode::Terminal() const
{
    if (!Shape().terminal)
        throw std::logic_error("Non-terminal node has no terminal outcome");
    return *Shape().terminal;
}

bool GameNode::IsForcedRunout() const
{
    return Kind() == NodeKind::Chance && state_.HasAllInPlayer();
}

BettingEdge GameNode::GetBettingEdge(std::size_t index) const
{
    return {Shape().actions.at(index), NodeId(id_.Value() + static_cast<std::int32_t>(Shape().childOffsets.at(index)))};
}

std::size_t GameNode::ChanceOutcomeCount() const
{
    return Kind() == NodeKind::Chance ? 52 - state_.board.CardCount() : 0;
}

ChanceOutcome GameNode::GetChanceOutcome(std::size_t index) const
{
    if (index >= ChanceOutcomeCount())
        throw std::out_of_range("Chance outcome index");
    std::size_t ordinal = 0;
    for (int card = 0; card < 52; ++card)
    {
        if (core::Contains(state_.board, core::Card(card)))
            continue;
        if (ordinal++ == index)
        {
            const auto span = (*topology_)[Shape().children.front()].logicalNodes;
            return {core::Card(card), NodeId(id_.Value() + 1 + static_cast<std::int32_t>(index * span))};
        }
    }
    throw std::logic_error("Missing chance outcome");
}

GameNode GameNode::Child(std::size_t index) const
{
    if (Kind() == NodeKind::Chance)
    {
        const auto outcome = GetChanceOutcome(index);
        return {topology_, Shape().children.front(), outcome.NextNode(), state_.board.Append(outcome.DealtCard()), ParentEdge{id_, index}};
    }
    const auto edge = GetBettingEdge(index);
    return {topology_, Shape().children.at(index), edge.NextNode(), state_.board, ParentEdge{id_, index}};
}

CompiledGame::CompiledGame(GameSpec spec, std::vector<BettingTopologyNode> topology)
    : spec_(std::move(spec)), topology_(std::move(topology)), showdownRanks_(kRunoutCount * kHandCount, kBlockedRank)
{
    std::vector<core::HoleCards> hands;
    hands.reserve(kHandCount);
    for (int high = 1; high < 52; ++high)
        for (int low = 0; low < high; ++low)
            hands.emplace_back(core::Card(low), core::Card(high));

    std::size_t rowOffset = 0;
    for (int high = 1; high < 52; ++high)
    {
        const core::Card first(high);
        if (core::Contains(spec_.initialBoard, first))
            continue;
        for (int low = 0; low < high; ++low)
        {
            const core::Card second(low);
            if (core::Contains(spec_.initialBoard, second))
                continue;

            // Only the rank row is shared by reversed runouts; game histories keep their original order.
            const core::Board board = spec_.initialBoard.Append(first).Append(second);
            showdownRowOffsets_[core::CardPairIndex(first, second)] = rowOffset;
            for (std::size_t handIndex = 0; handIndex < hands.size(); ++handIndex)
            {
                if (!core::Overlaps(hands[handIndex], board))
                    showdownRanks_[rowOffset + handIndex] = static_cast<std::uint16_t>(core::EvaluateHoldem(hands[handIndex], board));
            }
            rowOffset += kHandCount;
        }
    }
}

int CompiledGame::ShowdownRank(const core::Board& board, core::HoleCards hand) const
{
    const std::size_t rowOffset = showdownRowOffsets_[core::CardPairIndex(board.CardAt(3), board.CardAt(4))];
    const auto cards = hand.Cards();
    const std::uint16_t rank = showdownRanks_[rowOffset + core::CardPairIndex(cards[0], cards[1])];
    if (rank == kBlockedRank)
        throw std::runtime_error("Private hand overlaps the public board");
    return rank;
}

GameNode CompiledGame::GetNode(NodeId id) const
{
    if (id.Value() < 0 || static_cast<std::size_t>(id.Value()) >= NodeCount())
        throw std::out_of_range("Node ID does not belong to this game");
    GameNode node(&topology_, 0, Root(), spec_.initialBoard, std::nullopt);
    while (node.Id() != id)
    {
        const auto& shape = node.Shape();
        const auto relative = static_cast<std::uint32_t>(id.Value() - node.Id().Value());
        std::size_t edge;
        if (node.Kind() == NodeKind::Chance)
            edge = (relative - 1) / topology_[shape.children.front()].logicalNodes;
        else
            edge = static_cast<std::size_t>(
                std::upper_bound(shape.childOffsets.begin(), shape.childOffsets.end(), relative) - shape.childOffsets.begin() - 1
            );
        node = node.Child(edge);
    }
    return node;
}

GameTreeSize CompiledGame::Size() const
{
    std::size_t bytes =
        sizeof(*this) + topology_.capacity() * sizeof(BettingTopologyNode) + showdownRanks_.capacity() * sizeof(std::uint16_t);
    for (const auto& node : topology_)
        bytes += node.actions.capacity() * sizeof(BettingAction) +
                 (node.children.capacity() + node.childOffsets.capacity()) * sizeof(std::uint32_t);
    const auto& root = topology_.front();
    return {
        root.logicalNodes, topology_.size(), root.traversalNodes, root.decisionNodes, root.actionEntries, root.depth, root.maxActions, bytes
    };
}

} // namespace solver::game
