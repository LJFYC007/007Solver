#pragma once

#include "game/BettingAction.h"
#include "game/PublicState.h"
#include <optional>
#include <vector>

namespace solver::game
{
struct AggressiveActionRange
{
    BettingActionKind kind;
    core::Chips passiveAmountTo;
    core::Chips minimumAmountTo;
    core::Chips maximumAmountTo;
};

struct LegalActionSet
{
    std::vector<BettingAction> passiveActions;
    std::optional<AggressiveActionRange> aggression;
};

enum class ActionTransition : std::uint8_t
{
    Continue,
    BettingRoundComplete,
    Fold,
};

struct ActionApplication
{
    PublicState state;
    ActionTransition transition = ActionTransition::Continue;
};

bool IsAllIn(const PublicState& state, const BettingAction& action);

LegalActionSet GetLegalActions(const PublicState& state);
ActionApplication ApplyAction(const PublicState& state, const BettingAction& action);
core::Chips ChipsCommitted(const PublicState& state, const BettingAction& action);
} // namespace solver::game
