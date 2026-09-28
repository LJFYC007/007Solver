#pragma once

#include "core/Chips.h"
#include "core/PokerTypes.h"
#include "game/BettingAction.h"
#include "game/CompiledGame.h"
#include "game/Identifiers.h"
#include <array>
#include <optional>
#include <vector>

namespace solver::analysis
{
struct NodeStateReport
{
    core::Street street;
    core::Board board;
    core::Chips pot;
    std::array<core::Chips, 2> stacks;
    std::array<float, 2> rangeCombos{};
};

struct HandReport
{
    core::HoleCards cards;
    float inputRangeWeight;
    float ownReachWeight;
    float marginalReachMass;
    std::optional<float> nodeStrategyEv;
    std::vector<float> strategy;
    // EV of each action followed by the node strategy; empty exactly when nodeStrategyEv is.
    std::vector<float> actionEvs;
};

struct ActionReport
{
    game::BettingActionKind kind;
    bool isAllIn;
    core::Chips amountTo;
    core::Chips chipsCommitted;
    game::NodeId nextNodeId;
};

struct ChanceOutcomeReport
{
    core::Card card;
    game::NodeId nextNodeId;
};

// The node strategy EVs of the player waiting at a decision, which a node report omits because they
// need another evaluation of the subtree.
struct OpponentHandReport
{
    core::HoleCards cards;
    float marginalReachMass;
    std::optional<float> nodeStrategyEv;
};

struct OpponentEvReport
{
    game::NodeId nodeId;
    core::PlayerId player;
    std::vector<OpponentHandReport> hands;
};

struct NodeReport
{
    game::NodeId nodeId;
    game::NodeKind kind;
    NodeStateReport state;
    std::optional<core::PlayerId> actor;
    std::vector<HandReport> hands;
    std::vector<ActionReport> actions;
    std::vector<ChanceOutcomeReport> outcomes;
    std::optional<game::TerminalOutcome> terminal;
};
} // namespace solver::analysis
