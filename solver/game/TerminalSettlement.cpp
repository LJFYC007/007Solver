#include "game/TerminalSettlement.h"

namespace solver::game
{
float TerminalSettlement::NetPayoffFromStart(core::PlayerId player) const
{
    const core::Chips contribution = contributionsSinceStart[player.Index()];
    if (!winner.has_value())
    {
        const core::Chips proceedsBeforeSplit = core::Chips::FromRaw(grossPot.Raw() - 2 * contribution.Raw());
        return core::ToChipUnits(proceedsBeforeSplit) / 2.0f;
    }
    if (*winner == player)
        return core::ToChipUnits(grossPot - contribution);
    return -core::ToChipUnits(contribution);
}

} // namespace solver::game
