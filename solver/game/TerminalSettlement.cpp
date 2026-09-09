#include "game/TerminalSettlement.h"
#include "game/CompiledGame.h"
#include <stdexcept>

namespace solver::game
{
namespace
{
float ToChipUnits(core::Chips chips)
{
    return static_cast<float>(chips.Raw()) / static_cast<float>(core::Chips::kUnitsPerChip);
}
} // namespace

float TerminalSettlement::NetPayoffFromStart(core::PlayerId player) const
{
    const core::Chips contribution = contributionsSinceStart[player.Index()];
    if (!winner.has_value())
    {
        const core::Chips proceedsBeforeSplit = core::Chips::FromRaw(grossPot.Raw() - 2 * contribution.Raw());
        return ToChipUnits(proceedsBeforeSplit) / 2.0f;
    }
    if (*winner == player)
        return ToChipUnits(grossPot - contribution);
    return -ToChipUnits(contribution);
}

TerminalSettlement CompiledGame::CalculateTerminalSettlement(
    NodeId startNode,
    NodeId terminalNode,
    core::HoleCards player0Hand,
    core::HoleCards player1Hand
) const
{
    const GameNode& start = GetNode(startNode);
    const GameNode& terminal = GetNode(terminalNode);
    if (terminal.Kind() != NodeKind::Terminal)
        throw std::invalid_argument("Terminal settlement requested for a non-terminal node");

    const std::array<core::Chips, 2> contributionsSinceStart{
        start.State().stacks[0] - terminal.State().stacks[0],
        start.State().stacks[1] - terminal.State().stacks[1],
    };
    if (terminal.Terminal().kind == TerminalKind::Fold)
    {
        const core::PlayerId foldedPlayer = *terminal.Terminal().foldedPlayer;
        return {terminal.State().pot, contributionsSinceStart, foldedPlayer.Other()};
    }

    const int player0Rank = ShowdownRank(terminal.State().board, player0Hand);
    const int player1Rank = ShowdownRank(terminal.State().board, player1Hand);
    if (player0Rank > player1Rank)
        return {terminal.State().pot, contributionsSinceStart, core::PlayerId::Player0()};
    if (player0Rank < player1Rank)
        return {terminal.State().pot, contributionsSinceStart, core::PlayerId::Player1()};
    return {terminal.State().pot, contributionsSinceStart, std::nullopt};
}
} // namespace solver::game
