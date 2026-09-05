#include "game/CompiledGame.h"
#include "game/TerminalSettlement.h"
#include <stdexcept>
#include <utility>

namespace solver::game
{
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

CompiledGame::CompiledGame(GameSpec spec, std::vector<GameNode> nodes) : spec_(std::move(spec)), nodes_(std::move(nodes)) {}

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
    const TerminalSettlement settlement = CalculateTerminalSettlement(GetNode(Root()), GetNode(terminalNode), player0Hand, player1Hand);
    const float initialPot = static_cast<float>(spec_.initialPot.Raw()) / static_cast<float>(core::Chips::kUnitsPerChip);
    const float player0Value = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - initialPot / 2.0f;
    return {player0Value, -player0Value};
}
} // namespace solver::game
