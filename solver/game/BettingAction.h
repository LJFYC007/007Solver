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
    BettingAction(BettingActionKind kind, core::Chips amountTo) : kind_(kind), amountTo_(amountTo) {}

    BettingActionKind Kind() const { return kind_; }
    core::Chips AmountTo() const { return amountTo_; }

private:
    BettingActionKind kind_;
    core::Chips amountTo_;
};
} // namespace solver::game
