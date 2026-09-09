#include "game/CompiledGame.h"
#include "core/Evaluator.h"
#include "game/TerminalSettlement.h"
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

std::size_t CardPairIndex(core::Card first, core::Card second)
{
    const int low = std::min(first.Index(), second.Index());
    const int high = std::max(first.Index(), second.Index());
    return static_cast<std::size_t>(high * (high - 1) / 2 + low);
}
} // namespace

GameNode::GameNode(NodeKind kind, PublicState state, std::optional<ParentEdge> parent, std::optional<TerminalOutcome> terminal)
    : kind_(kind), state_(std::move(state)), parent_(parent), terminal_(terminal)
{}

const TerminalOutcome& GameNode::Terminal() const
{
    if (!terminal_)
        throw std::logic_error("Non-terminal node has no terminal outcome");
    return *terminal_;
}

const BettingEdge& GameNode::GetBettingEdge(std::size_t index) const
{
    return bettingEdges_.at(index);
}

const ChanceOutcome& GameNode::GetChanceOutcome(std::size_t index) const
{
    return chanceOutcomes_.at(index);
}

CompiledGame::CompiledGame(GameSpec spec, std::vector<GameNode> nodes)
    : spec_(std::move(spec)), nodes_(std::move(nodes)), showdownRanks_(kRunoutCount * kHandCount, kBlockedRank)
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
            showdownRowOffsets_[CardPairIndex(first, second)] = rowOffset;
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
    const std::size_t rowOffset = showdownRowOffsets_[CardPairIndex(board.CardAt(3), board.CardAt(4))];
    const auto cards = hand.Cards();
    const std::uint16_t rank = showdownRanks_[rowOffset + CardPairIndex(cards[0], cards[1])];
    if (rank == kBlockedRank)
        throw std::runtime_error("Private hand overlaps the public board");
    return rank;
}

int CompiledGame::ShowdownRank(NodeId terminalNode, core::HoleCards hand) const
{
    const GameNode& node = GetNode(terminalNode);
    if (node.Kind() != NodeKind::Terminal || node.Terminal().kind != TerminalKind::Showdown)
        throw std::invalid_argument("Showdown rank requires a showdown terminal");
    return ShowdownRank(node.State().board, hand);
}

const GameNode& CompiledGame::GetNode(NodeId id) const
{
    return nodes_.at(static_cast<std::size_t>(id.Value()));
}

std::pair<float, float> CompiledGame::CalculateZeroSumUtility(
    NodeId terminalNode,
    core::HoleCards player0Hand,
    core::HoleCards player1Hand
) const
{
    const TerminalSettlement settlement = CalculateTerminalSettlement(Root(), terminalNode, player0Hand, player1Hand);
    const float initialPot = static_cast<float>(spec_.initialPot.Raw()) / static_cast<float>(core::Chips::kUnitsPerChip);
    const float player0Value = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - initialPot / 2.0f;
    return {player0Value, -player0Value};
}
} // namespace solver::game
