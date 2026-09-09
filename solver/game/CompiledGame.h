#pragma once

#include "game/BettingAction.h"
#include "game/GameSpec.h"
#include "game/Identifiers.h"
#include "game/PublicState.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace solver::game
{
class GameTreeBuilder;
struct TerminalSettlement;

enum class NodeKind : std::uint8_t
{
    Decision,
    Chance,
    Terminal,
};

enum class TerminalKind : std::uint8_t
{
    Fold,
    Showdown,
};

struct TerminalOutcome
{
    TerminalKind kind;
    std::optional<core::PlayerId> foldedPlayer;
};

struct ParentEdge
{
    NodeId node;
    std::size_t edgeIndex;
};

class BettingEdge
{
public:
    BettingEdge(BettingAction action, NodeId nextNode) : action_(action), nextNode_(nextNode) {}

    const BettingAction& Action() const { return action_; }
    NodeId NextNode() const { return nextNode_; }

private:
    BettingAction action_;
    NodeId nextNode_;
};

class ChanceOutcome
{
public:
    ChanceOutcome(core::Card dealtCard, NodeId nextNode) : dealtCard_(dealtCard), nextNode_(nextNode) {}

    core::Card DealtCard() const { return dealtCard_; }
    NodeId NextNode() const { return nextNode_; }

private:
    core::Card dealtCard_;
    NodeId nextNode_;
};

class GameNode
{
public:
    NodeKind Kind() const { return kind_; }
    const PublicState& State() const { return state_; }
    const std::optional<ParentEdge>& Parent() const { return parent_; }
    const TerminalOutcome& Terminal() const;

    std::size_t BettingEdgeCount() const { return bettingEdges_.size(); }
    const BettingEdge& GetBettingEdge(std::size_t index) const;
    std::size_t ChanceOutcomeCount() const { return chanceOutcomes_.size(); }
    const ChanceOutcome& GetChanceOutcome(std::size_t index) const;

private:
    friend class GameTreeBuilder;

    GameNode(NodeKind kind, PublicState state, std::optional<ParentEdge> parent, std::optional<TerminalOutcome> terminal);

    NodeKind kind_;
    PublicState state_;
    std::optional<ParentEdge> parent_;
    std::optional<TerminalOutcome> terminal_;
    std::vector<BettingEdge> bettingEdges_;
    std::vector<ChanceOutcome> chanceOutcomes_;
};

class CompiledGame
{
public:
    const GameSpec& Spec() const { return spec_; }
    NodeId Root() const { return NodeId(0); }
    std::size_t NodeCount() const { return nodes_.size(); }
    const GameNode& GetNode(NodeId id) const;

    // Cached rank at a showdown terminal; larger ranks are stronger.
    int ShowdownRank(NodeId terminalNode, core::HoleCards hand) const;

    TerminalSettlement CalculateTerminalSettlement(
        NodeId startNode,
        NodeId terminalNode,
        core::HoleCards player0Hand,
        core::HoleCards player1Hand
    ) const;
    std::pair<float, float> CalculateZeroSumUtility(NodeId terminalNode, core::HoleCards player0Hand, core::HoleCards player1Hand) const;

private:
    friend std::shared_ptr<const CompiledGame> CompileGame(const GameSpec& gameSpec);

    CompiledGame(GameSpec spec, std::vector<GameNode> nodes);
    int ShowdownRank(const core::Board& board, core::HoleCards hand) const;

    GameSpec spec_;
    std::vector<GameNode> nodes_;
    std::array<std::size_t, 1326> showdownRowOffsets_{};
    std::vector<std::uint16_t> showdownRanks_;
};
} // namespace solver::game
