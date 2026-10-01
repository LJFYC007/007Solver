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
    // Rake is capped in scenario chip units; either zero disables it.
    float rakePercent = 0.0f;
    float rakeCap = 0.0f;

    bool HasRake() const { return rakePercent > 0.0f && rakeCap > 0.0f; }
};
} // namespace solver::game
