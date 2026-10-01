#include "game/TerminalSettlement.h"
#include "game/GameSpec.h"
#include "game/PublicState.h"
#include <algorithm>

namespace solver::game
{
float TerminalRake(const GameSpec& spec, const PublicState& state)
{
    const auto unmatched = std::max(state.streetContributions[0], state.streetContributions[1]) -
                           std::min(state.streetContributions[0], state.streetContributions[1]);
    return std::min(core::ToChipUnits(state.pot - unmatched) * (spec.rakePercent / 100.0f), spec.rakeCap);
}

float TerminalSettlement::NetPayoffFromStart(core::PlayerId player) const
{
    const core::Chips contribution = contributionsSinceStart[player.Index()];
    if (!winner.has_value())
    {
        const core::Chips proceedsBeforeSplit = core::Chips::FromRaw(grossPot.Raw() - 2 * contribution.Raw());
        return (core::ToChipUnits(proceedsBeforeSplit) - rake) / 2.0f;
    }
    if (*winner == player)
        return core::ToChipUnits(grossPot - contribution) - rake;
    return -core::ToChipUnits(contribution);
}

} // namespace solver::game
