#include "game/BettingAbstraction.h"
#include "game/BettingRules.h"
#include "game/PublicState.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace solver::game
{
namespace
{
core::Chips RoundHalfUp(core::Chips value, std::int64_t numerator, std::int64_t denominator)
{
    if (value < core::Chips{} || numerator < 0 || denominator <= 0)
        throw std::invalid_argument("Bet fractions require non-negative values and a positive denominator");

    // Bet fractions are rounded to the nearest tenth of a chip, with exact halves rounded up.
    if (value.Raw() != 0 && numerator > (std::numeric_limits<std::int64_t>::max() - denominator / 2) / value.Raw())
        throw std::invalid_argument("Bet fraction exceeds the supported chip range");
    const std::int64_t scaled = static_cast<std::int64_t>(value.Raw()) * numerator;
    const std::int64_t rounded = (scaled + denominator / 2) / denominator;
    if (rounded > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("Bet fraction exceeds the supported chip range");
    return core::Chips::FromRaw(static_cast<std::int32_t>(rounded));
}

core::Chips AddSizingAmount(core::Chips contribution, core::Chips additionalChips)
{
    const std::int64_t amountTo = static_cast<std::int64_t>(contribution.Raw()) + additionalChips.Raw();
    if (amountTo > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("Bet size exceeds the supported chip range");
    return core::Chips::FromRaw(static_cast<std::int32_t>(amountTo));
}

core::Chips ResolveBetAmountTo(const BetSize& size, const PublicState& state)
{
    if (size.kind == BetSizeKind::PotFractionOfCurrentPot)
        return AddSizingAmount(
            state.Contribution(state.playerToAct), RoundHalfUp(state.pot, size.fractionNumerator, size.fractionDenominator)
        );
    if (size.absoluteAmountTo < core::Chips{})
        throw std::invalid_argument("Absolute bet amount-to cannot be negative");
    return size.absoluteAmountTo;
}

core::Chips ResolveRaiseAmountTo(const RaiseSize& size, const PublicState& state, core::Chips callAmountTo)
{
    if (size.kind == RaiseSizeKind::PotFractionOfPotAfterCallAsRaiseBy)
    {
        const core::Chips callAmount = callAmountTo - state.Contribution(state.playerToAct);
        const core::Chips potAfterCall = state.pot + callAmount;
        return AddSizingAmount(callAmountTo, RoundHalfUp(potAfterCall, size.fractionNumerator, size.fractionDenominator));
    }
    if (size.absoluteAmountTo < core::Chips{})
        throw std::invalid_argument("Absolute raise amount-to cannot be negative");
    return size.absoluteAmountTo;
}
} // namespace

std::vector<BettingAction> BettingAbstraction::SelectActions(const PublicState& state, const LegalActionSet& legalActions) const
{
    std::vector<BettingAction> actions = legalActions.passiveActions;
    if (!legalActions.aggression.has_value())
        return actions;

    const AggressiveActionRange& range = *legalActions.aggression;
    if (range.kind == BettingActionKind::Raise && state.raiseCount >= maxRaises)
        return actions;
    const auto& sizes = streets.at(static_cast<std::size_t>(state.street));
    const auto addAmountTo = [&](core::Chips amountTo)
    {
        // A configured size always yields legal aggression, including a short all-in.
        amountTo = std::min(range.maximumAmountTo, std::max(range.minimumAmountTo, amountTo));

        const std::int64_t potAfterCall = static_cast<std::int64_t>(state.pot.Raw()) + 2LL * amountTo.Raw() -
                                          state.Contribution(state.playerToAct).Raw() - state.Contribution(state.playerToAct.Other()).Raw();
        const auto remaining = range.maximumAmountTo.Raw() - amountTo.Raw();
        if (allInSpr > 0.0f && static_cast<float>(remaining) / potAfterCall <= allInSpr)
            amountTo = range.maximumAmountTo;

        const bool duplicate =
            std::any_of(actions.begin(), actions.end(), [&](const BettingAction& action) { return action.AmountTo() == amountTo; });
        if (!duplicate)
            actions.emplace_back(range.kind, amountTo);
    };

    if (range.kind == BettingActionKind::Bet)
    {
        for (const BetSize& size : sizes.betSizes)
            addAmountTo(ResolveBetAmountTo(size, state));
    }
    else
    {
        for (const RaiseSize& size : sizes.raiseSizes)
            addAmountTo(ResolveRaiseAmountTo(size, state, range.passiveAmountTo));
    }

    return actions;
}
} // namespace solver::game
