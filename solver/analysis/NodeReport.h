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
