#include "game/GameCompiler.h"
#include "game/BettingRules.h"
#include "game/CompiledGame.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace solver::game
{
class GameTreeBuilder
{
public:
    explicit GameTreeBuilder(const GameSpec& spec) : spec_(spec) {}

    std::vector<GameNode> Build()
    {
        PublicState rootState{
            spec_.outOfPositionPlayer,
            spec_.initialPot,
            spec_.initialStacks,
            {core::Chips{}, core::Chips{}},
            spec_.initialStreet,
            spec_.initialBoard,
        };
        const bool allIn = rootState.stacks[0] == core::Chips{} || rootState.stacks[1] == core::Chips{};
        BuildNode(allIn ? NodeKind::Chance : NodeKind::Decision, std::move(rootState), std::nullopt, std::nullopt);
        return std::move(nodes_);
    }

private:
    const GameSpec& spec_;
    std::vector<GameNode> nodes_;

    NodeId BuildNode(NodeKind kind, PublicState state, std::optional<ParentEdge> parent, std::optional<TerminalOutcome> terminal)
    {
        if (kind == NodeKind::Chance && state.street == core::Street::River)
        {
            kind = NodeKind::Terminal;
            terminal = TerminalOutcome{TerminalKind::Showdown, std::nullopt};
        }
        if (nodes_.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::runtime_error("Game tree exceeds the supported node count");

        // Every action and dealt card creates a distinct child, preserving the full observable history.
        const NodeId nodeId(static_cast<std::int32_t>(nodes_.size()));
        nodes_.push_back(GameNode(kind, state, parent, terminal));
        if (kind == NodeKind::Chance)
            BuildChanceOutcomes(nodeId, state);
        else if (kind == NodeKind::Decision)
            BuildBettingEdges(nodeId, state);
        return nodeId;
    }

    void BuildChanceOutcomes(NodeId nodeId, const PublicState& state)
    {
        const core::Street nextStreet = state.street == core::Street::Flop ? core::Street::Turn : core::Street::River;
        const bool allIn = state.stacks[0] == core::Chips{} || state.stacks[1] == core::Chips{};
        for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
        {
            const core::Card card(cardIndex);
            if (core::Contains(state.board, card))
                continue;

            PublicState childState{
                spec_.outOfPositionPlayer,
                state.pot,
                state.stacks,
                {core::Chips{}, core::Chips{}},
                nextStreet,
                state.board.Append(card),
            };
            const ParentEdge parent{nodeId, nodes_[nodeId.Value()].ChanceOutcomeCount()};
            const NodeId childId = BuildNode(allIn ? NodeKind::Chance : NodeKind::Decision, std::move(childState), parent, std::nullopt);
            nodes_[nodeId.Value()].chanceOutcomes_.emplace_back(card, childId);
        }
    }

    void BuildBettingEdges(NodeId nodeId, const PublicState& state)
    {
        const std::vector<BettingAction> actions = spec_.bettingAbstraction.SelectActions(state, GetLegalActions(state));
        for (const BettingAction& action : actions)
        {
            ActionApplication application = ApplyAction(state, action);
            NodeKind childKind = NodeKind::Decision;
            std::optional<TerminalOutcome> terminal;
            if (application.transition == ActionTransition::Fold)
            {
                childKind = NodeKind::Terminal;
                terminal = TerminalOutcome{TerminalKind::Fold, state.playerToAct};
            }
            else if (application.transition == ActionTransition::BettingRoundComplete)
            {
                childKind = NodeKind::Chance;
            }

            const ParentEdge parent{nodeId, nodes_[nodeId.Value()].BettingEdgeCount()};
            const NodeId childId = BuildNode(childKind, std::move(application.state), parent, terminal);
            nodes_[nodeId.Value()].bettingEdges_.emplace_back(action, childId);
        }
    }
};

std::shared_ptr<const CompiledGame> CompileGame(const GameSpec& gameSpec)
{
    if (gameSpec.initialStreet != core::Street::Flop || gameSpec.initialBoard.CardCount() != 3)
        throw std::invalid_argument("The game compiler requires a three-card flop");
    if (gameSpec.initialPot < core::Chips{} || gameSpec.initialStacks[0] < core::Chips{} || gameSpec.initialStacks[1] < core::Chips{})
        throw std::invalid_argument("Game pot and stacks cannot be negative");
    const std::int64_t maximumPot = static_cast<std::int64_t>(gameSpec.initialPot.Raw()) +
                                    2 * static_cast<std::int64_t>(std::min(gameSpec.initialStacks[0], gameSpec.initialStacks[1]).Raw());
    if (maximumPot > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("Game pot can exceed the supported chip range");

    return std::shared_ptr<const CompiledGame>(new CompiledGame(gameSpec, GameTreeBuilder(gameSpec).Build()));
}
} // namespace solver::game
