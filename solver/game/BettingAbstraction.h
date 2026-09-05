#pragma once

#include "core/Chips.h"
#include "game/BettingAction.h"
#include <cstdint>
#include <vector>

namespace solver::game
{
struct PublicState;
struct LegalActionSet;

enum class BetSizeKind : std::uint8_t
{
    PotFractionOfCurrentPot,
    AbsoluteAmountTo,
};

struct BetSize
{
    BetSizeKind kind = BetSizeKind::PotFractionOfCurrentPot;
    std::int64_t fractionNumerator = 0;
    std::int64_t fractionDenominator = 1;
    core::Chips absoluteAmountTo;
};

enum class RaiseSizeKind : std::uint8_t
{
    PotFractionOfPotAfterCallAsRaiseBy,
    AbsoluteAmountTo,
};

struct RaiseSize
{
    RaiseSizeKind kind = RaiseSizeKind::PotFractionOfPotAfterCallAsRaiseBy;
    std::int64_t fractionNumerator = 0;
    std::int64_t fractionDenominator = 1;
    core::Chips absoluteAmountTo;
};

struct BettingAbstraction
{
    std::vector<BetSize> betSizes;
    std::vector<RaiseSize> raiseSizes;
    // Append the effective-stack maximum in addition to explicitly configured sizes.
    bool includeMaximumBet = true;
    bool includeMaximumRaise = true;

    static BettingAbstraction Default();

    std::vector<BettingAction> SelectActions(const PublicState& state, const LegalActionSet& legalActions) const;
};
} // namespace solver::game
