#pragma once

#include "core/Chips.h"
#include "core/PokerTypes.h"
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

} // namespace solver::game
