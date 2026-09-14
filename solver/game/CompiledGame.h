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
struct TerminalSettlement;
enum class NodeKind : std::uint8_t
{
    Decision,
    Chance,
    Terminal
};
enum class TerminalKind : std::uint8_t
{
    Fold,
    Showdown
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
    ChanceOutcome(core::Card card, NodeId nextNode) : card_(card), nextNode_(nextNode) {}
    core::Card DealtCard() const { return card_; }
    NodeId NextNode() const { return nextNode_; }

private:
    core::Card card_;
    NodeId nextNode_;
};
// Board-independent betting history. Chance has one continuation template, reused for
// every concrete card. Subtree spans assign distinct preorder IDs to those histories.
struct BettingTopologyNode
{
    NodeKind kind;
    PublicState state;
    std::optional<TerminalOutcome> terminal;
    std::vector<BettingAction> actions;
    std::vector<std::uint32_t> children;
    std::vector<std::uint32_t> childOffsets;
    std::uint32_t logicalNodes = 1;
    std::uint32_t traversalNodes = 1;
    std::array<std::uint64_t, 2> decisionNodes{};
    std::array<std::uint64_t, 2> actionEntries{};
    std::uint32_t depth = 1;
    std::uint32_t maxActions = 1;
};
// A small value view. State, parent and child IDs identify one concrete history;
// only immutable betting topology is shared. The game must outlive the view.
class GameNode
{
public:
    NodeId Id() const { return id_; }
    NodeKind Kind() const { return Shape().kind; }
    const PublicState& State() const { return state_; }
    const std::optional<ParentEdge>& Parent() const { return parent_; }
    const TerminalOutcome& Terminal() const;
    bool IsForcedRunout() const;
    std::size_t SubtreeNodeCount() const { return Shape().logicalNodes; }
    std::size_t TraversalNodeCount() const { return Shape().traversalNodes; }
    std::size_t BettingEdgeCount() const { return Shape().actions.size(); }
    BettingEdge GetBettingEdge(std::size_t index) const;
    std::size_t ChanceOutcomeCount() const;
    ChanceOutcome GetChanceOutcome(std::size_t index) const;
    GameNode Child(std::size_t index) const;

private:
    friend class CompiledGame;
    GameNode(
        const std::vector<BettingTopologyNode>* topology,
        std::uint32_t shape,
        NodeId id,
        core::Board board,
        std::optional<ParentEdge> parent
    );
    const BettingTopologyNode& Shape() const { return (*topology_)[shape_]; }
    const std::vector<BettingTopologyNode>* topology_;
    std::uint32_t shape_;
    NodeId id_;
    PublicState state_;
    std::optional<ParentEdge> parent_;
};
struct GameTreeSize
{
    std::size_t logicalNodes;
    std::size_t topologyNodes;
    std::size_t traversalNodes;
    std::array<std::uint64_t, 2> decisionNodes;
    std::array<std::uint64_t, 2> actionEntries;
    std::size_t depth;
    std::size_t maxActions;
    std::size_t storageBytes;
};
class CompiledGame
{
public:
    const GameSpec& Spec() const { return spec_; }
    NodeId Root() const { return NodeId(0); }
    std::size_t NodeCount() const { return topology_.front().logicalNodes; }
    GameNode GetNode(NodeId id) const;
    GameTreeSize Size() const;
    int ShowdownRank(NodeId terminalNode, core::HoleCards hand) const;
    int ShowdownRank(const core::Board& board, core::HoleCards hand) const;
    TerminalSettlement CalculateTerminalSettlement(
        NodeId startNode,
        NodeId terminalNode,
        core::HoleCards player0Hand,
        core::HoleCards player1Hand
    ) const;
    std::pair<float, float> CalculateZeroSumUtility(NodeId terminalNode, core::HoleCards player0Hand, core::HoleCards player1Hand) const;

private:
    friend std::shared_ptr<const CompiledGame> CompileGame(const GameSpec& gameSpec);
    CompiledGame(GameSpec spec, std::vector<BettingTopologyNode> topology);
    GameSpec spec_;
    std::vector<BettingTopologyNode> topology_;
    std::array<std::size_t, 1326> showdownRowOffsets_{};
    std::vector<std::uint16_t> showdownRanks_;
};
} // namespace solver::game
