#include "game/TerminalSettlement.h"

namespace solver::game
{
namespace
{
float ToChipUnits(core::Chips chips)
{
    return static_cast<float>(chips.Raw()) / static_cast<float>(core::Chips::kUnitsPerChip);
}
} // namespace

float TerminalSettlement::NetPayoffFromStart(core::PlayerId player) const
{
    const core::Chips contribution = contributionsSinceStart[player.Index()];
    if (!winner.has_value())
    {
        const core::Chips proceedsBeforeSplit = core::Chips::FromRaw(grossPot.Raw() - 2 * contribution.Raw());
        return ToChipUnits(proceedsBeforeSplit) / 2.0f;
    }
    if (*winner == player)
        return ToChipUnits(grossPot - contribution);
    return -ToChipUnits(contribution);
}

} // namespace solver::game
