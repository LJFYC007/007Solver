#pragma once

#include "core/Card.h"
#include "core/Chips.h"
#include "core/PokerTypes.h"
#include "game/CompiledGame.h"
#include <array>
#include <optional>
#include <utility>

namespace solver::game
{
struct TerminalSettlement
{
    core::Chips grossPot;
    std::array<core::Chips, 2> contributionsSinceStart;
    std::optional<core::PlayerId> winner;

    float NetPayoffFromStart(core::PlayerId player) const;
};

TerminalSettlement CalculateTerminalSettlement(
    const GameNode& start,
    const GameNode& terminal,
    core::HoleCards player0Hand,
    core::HoleCards player1Hand
);

} // namespace solver::game
