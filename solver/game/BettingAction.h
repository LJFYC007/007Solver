#pragma once

#include "core/Chips.h"
#include <cstdint>

namespace solver::game
{
enum class BettingActionKind : std::uint8_t
{
    Fold,
    Check,
    Call,
    Bet,
    Raise,
};

class BettingAction
{
public:
    BettingAction(BettingActionKind kind, core::Chips amountTo, float potPercent = 0.0f)
        : kind_(kind), amountTo_(amountTo), potPercent_(potPercent)
    {}

    BettingActionKind Kind() const { return kind_; }
    core::Chips AmountTo() const { return amountTo_; }
    // A bet's percentage of the pot, or a raise's of the pot after calling: the configured size, or the
    // size the legal limits made of it. Zero for passive actions.
    float PotPercent() const { return potPercent_; }

private:
    BettingActionKind kind_;
    core::Chips amountTo_;
    float potPercent_;
};
} // namespace solver::game
