#include "game/GameCompiler.h"
#include "game/BettingRules.h"
#include "game/CompiledGame.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace solver::game
{
class GameTreeBuilder
{
public:
    explicit GameTreeBuilder(const GameSpec& spec) : spec_(spec) {}
    std::vector<BettingTopologyNode> Build()
    {
        PublicState root{
            spec_.outOfPositionPlayer,
            spec_.initialPot,
            spec_.initialStacks,
            {core::Chips{}, core::Chips{}},
            spec_.initialStreet,
            spec_.initialBoard
        };
        BuildNode(IsAllIn(root) ? NodeKind::Chance : NodeKind::Decision, root, std::nullopt);
        return std::move(nodes_);
    }

private:
    const GameSpec& spec_;
    std::vector<BettingTopologyNode> nodes_;
    static bool IsAllIn(const PublicState& state) { return state.stacks[0] == core::Chips{} || state.stacks[1] == core::Chips{}; }
    std::uint32_t BuildNode(NodeKind kind, PublicState state, std::optional<TerminalOutcome> terminal)
    {
        if (kind == NodeKind::Chance && state.street == core::Street::River)
        {
            kind = NodeKind::Terminal;
            terminal = TerminalOutcome{TerminalKind::Showdown, std::nullopt};
        }
        if (nodes_.size() >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::runtime_error("Betting topology exceeds the supported node count");
        const auto index = static_cast<std::uint32_t>(nodes_.size());
        nodes_.push_back({kind, state, terminal});
        if (kind == NodeKind::Chance)
        {
            // Legal betting does not depend on card ranks or suits.
            int card = 0;
            while (core::Contains(state.board, core::Card(card)))
                ++card;
            PublicState next{
                spec_.outOfPositionPlayer,
                state.pot,
                state.stacks,
                {core::Chips{}, core::Chips{}},
                state.street == core::Street::Flop ? core::Street::Turn : core::Street::River,
                state.board.Append(core::Card(card))
            };
            const auto child = BuildNode(IsAllIn(state) ? NodeKind::Chance : NodeKind::Decision, next, std::nullopt);
            nodes_[index].children.push_back(child);
        }
        else if (kind == NodeKind::Decision)
        {
            const auto actions = spec_.bettingAbstraction.SelectActions(state, GetLegalActions(state));
            for (const auto& action : actions)
            {
                const auto application = ApplyAction(state, action);
                NodeKind next = NodeKind::Decision;
                std::optional<TerminalOutcome> outcome;
                if (application.transition == ActionTransition::Fold)
                {
                    next = NodeKind::Terminal;
                    outcome = TerminalOutcome{TerminalKind::Fold, state.playerToAct};
                }
                else if (application.transition == ActionTransition::BettingRoundComplete)
                    next = NodeKind::Chance;
                const auto child = BuildNode(next, application.state, outcome);
                nodes_[index].actions.push_back(action);
                nodes_[index].children.push_back(child);
            }
        }
        auto& node = nodes_[index];
        const bool forced = kind == NodeKind::Chance && IsAllIn(state);
        const std::uint64_t copies = kind == NodeKind::Chance ? 52 - state.board.CardCount() : 1;
        std::uint64_t logical = 1;
        std::uint64_t active = 1;
        if (kind == NodeKind::Decision)
        {
            node.decisionNodes[state.playerToAct.Index()] = 1;
            node.actionEntries[state.playerToAct.Index()] = node.actions.size();
            node.maxActions = static_cast<std::uint32_t>(node.actions.size());
        }
        for (auto child : node.children)
        {
            node.childOffsets.push_back(static_cast<std::uint32_t>(logical));
            const auto& subtree = nodes_[child];
            logical += copies * subtree.logicalNodes;
            if (logical > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
                throw std::runtime_error("Game tree exceeds the supported logical node count; reduce betting branches");
            if (!forced)
            {
                active += copies * subtree.traversalNodes;
                node.depth = std::max(node.depth, subtree.depth + 1);
                node.maxActions = std::max(node.maxActions, subtree.maxActions);
                for (std::size_t player = 0; player < 2; ++player)
                {
                    node.decisionNodes[player] += copies * subtree.decisionNodes[player];
                    node.actionEntries[player] += copies * subtree.actionEntries[player];
                }
            }
        }
        node.logicalNodes = static_cast<std::uint32_t>(logical);
        node.traversalNodes = static_cast<std::uint32_t>(active);
        return index;
    }
};
std::shared_ptr<const CompiledGame> CompileGame(const GameSpec& spec)
{
    if (spec.initialStreet != core::Street::Flop || spec.initialBoard.CardCount() != 3)
        throw std::invalid_argument("The game compiler requires a three-card flop");
    if (spec.initialPot < core::Chips{} || spec.initialStacks[0] < core::Chips{} || spec.initialStacks[1] < core::Chips{})
        throw std::invalid_argument("Game pot and stacks cannot be negative");
    const std::int64_t maximumPot = static_cast<std::int64_t>(spec.initialPot.Raw()) +
                                    2 * static_cast<std::int64_t>(std::min(spec.initialStacks[0], spec.initialStacks[1]).Raw());
    if (maximumPot > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("Game pot can exceed the supported chip range");
    return std::shared_ptr<const CompiledGame>(new CompiledGame(spec, GameTreeBuilder(spec).Build()));
}
} // namespace solver::game
