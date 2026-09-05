#pragma once

#include "core/Card.h"
#include "core/Chips.h"
#include "core/PokerTypes.h"
#include <array>

namespace solver::game
{
struct PublicState
{
    core::PlayerId playerToAct;
    core::Chips pot;
    std::array<core::Chips, 2> stacks;
    std::array<core::Chips, 2> streetContributions;
    core::Street street;
    core::Board board;
    bool lastActionWasCheck = false;
    core::Chips lastFullRaiseSize;

    core::Chips Stack(core::PlayerId player) const { return stacks[player.Index()]; }
    core::Chips Contribution(core::PlayerId player) const { return streetContributions[player.Index()]; }
};

core::Chips AmountToCall(const PublicState& state);
} // namespace solver::game
