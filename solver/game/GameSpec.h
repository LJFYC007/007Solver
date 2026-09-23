#pragma once

#include "core/Card.h"
#include "core/Chips.h"
#include "core/PokerTypes.h"
#include "game/BettingAbstraction.h"
#include <array>

namespace solver::game
{
struct GameSpec
{
    // Solves start on the flop.
    core::Board initialBoard;
    core::Chips initialPot;
    std::array<core::Chips, 2> initialStacks;
    core::PlayerId outOfPositionPlayer;
    BettingAbstraction bettingAbstraction;
};
} // namespace solver::game
