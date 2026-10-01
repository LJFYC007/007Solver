#pragma once

#include "core/Chips.h"
#include "core/PokerTypes.h"
#include <array>
#include <optional>

namespace solver::game
{
struct GameSpec;
struct PublicState;

// The rake a terminal pays: rakePercent of the matched pot, capped at rakeCap. The difference
// between current-street contributions is an uncalled bet, returned without rake.
float TerminalRake(const GameSpec& spec, const PublicState& state);

struct TerminalSettlement
{
    core::Chips grossPot;
    std::array<core::Chips, 2> contributionsSinceStart;
    std::optional<core::PlayerId> winner;
    float rake = 0.0f;

    float NetPayoffFromStart(core::PlayerId player) const;
};

} // namespace solver::game
