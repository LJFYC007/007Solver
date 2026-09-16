#pragma once

#include "core/Chips.h"
#include "game/BettingAction.h"
#include <array>
#include <cstdint>
#include <utility>
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

struct StreetBettingSizes
{
    std::vector<BetSize> betSizes;
    std::vector<RaiseSize> raiseSizes;
};

struct BettingAbstraction
{
    BettingAbstraction(std::array<StreetBettingSizes, 3> streetSizes, std::uint32_t maximumRaises, double allInThreshold)
        : streets(std::move(streetSizes)), maxRaises(maximumRaises), allInSpr(allInThreshold)
    {}

    // Flop, turn, river; both players use the same sizes on each street.
    std::array<StreetBettingSizes, 3> streets;
    // The opening bet does not count. At the cap, only passive actions remain.
    std::uint32_t maxRaises;
    // Replace a configured size when the effective SPR after a call is at most this value.
    double allInSpr;

    std::vector<BettingAction> SelectActions(const PublicState& state, const LegalActionSet& legalActions) const;
};
} // namespace solver::game
