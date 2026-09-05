#include "game/BettingRules.h"
#include <algorithm>
#include <stdexcept>

namespace solver::game
{
namespace
{
bool SameAction(const BettingAction& left, const BettingAction& right)
{
    return left.Kind() == right.Kind() && left.AmountTo() == right.AmountTo();
}

std::int64_t TotalChips(const PublicState& state)
{
    return static_cast<std::int64_t>(state.pot.Raw()) + state.stacks[0].Raw() + state.stacks[1].Raw();
}
} // namespace

core::Chips AmountToCall(const PublicState& state)
{
    return std::max(core::Chips{}, state.Contribution(state.playerToAct.Other()) - state.Contribution(state.playerToAct));
}

bool IsAllIn(const PublicState& state, const BettingAction& action)
{
    if (action.Kind() == BettingActionKind::Fold || action.Kind() == BettingActionKind::Check)
        return false;
    if (action.AmountTo() < state.Contribution(state.playerToAct))
        return false;
    return action.AmountTo() - state.Contribution(state.playerToAct) == state.Stack(state.playerToAct);
}

LegalActionSet GetLegalActions(const PublicState& state)
{
    const core::PlayerId player = state.playerToAct;
    const core::PlayerId opponent = player.Other();
    const core::Chips contribution = state.Contribution(player);
    const core::Chips opponentContribution = state.Contribution(opponent);
    const core::Chips stack = state.Stack(player);
    const core::Chips opponentStack = state.Stack(opponent);
    const core::Chips maximumMatchedContribution = std::min(contribution + stack, opponentContribution + opponentStack);

    LegalActionSet actions;
    const core::Chips amountToCall = AmountToCall(state);
    if (amountToCall > core::Chips{})
    {
        actions.passiveActions.emplace_back(BettingActionKind::Fold, contribution);
        const core::Chips callAmount = std::min(stack, amountToCall);
        const core::Chips callAmountTo = contribution + callAmount;
        actions.passiveActions.emplace_back(BettingActionKind::Call, callAmountTo);

        if (maximumMatchedContribution > callAmountTo && maximumMatchedContribution > opponentContribution)
        {
            const core::Chips minimumRaiseSize = std::max(amountToCall, state.lastFullRaiseSize);
            actions.aggression = AggressiveActionRange{
                BettingActionKind::Raise,
                callAmountTo,
                opponentContribution + minimumRaiseSize,
                maximumMatchedContribution,
            };
        }
        return actions;
    }

    actions.passiveActions.emplace_back(BettingActionKind::Check, contribution);
    if (maximumMatchedContribution > contribution)
    {
        actions.aggression = AggressiveActionRange{
            BettingActionKind::Bet,
            contribution,
            contribution + core::Chips::FromRaw(1),
            maximumMatchedContribution,
        };
    }
    return actions;
}

ActionApplication ApplyAction(const PublicState& state, const BettingAction& action)
{
    const LegalActionSet legalActions = GetLegalActions(state);
    bool isLegal = std::any_of(
        legalActions.passiveActions.begin(),
        legalActions.passiveActions.end(),
        [&](const BettingAction& candidate) { return SameAction(candidate, action); }
    );
    if (!isLegal && legalActions.aggression.has_value() && action.Kind() == legalActions.aggression->kind)
    {
        const AggressiveActionRange& range = *legalActions.aggression;
        // The effective-stack cap may be below the minimum full raise.
        isLegal = action.AmountTo() <= range.maximumAmountTo && action.AmountTo() > range.passiveAmountTo &&
                  (action.AmountTo() >= range.minimumAmountTo || action.AmountTo() == range.maximumAmountTo);
    }
    if (!isLegal)
        throw std::invalid_argument("Betting action is not legal in the current public state");

    ActionApplication result{state};
    const core::PlayerId player = state.playerToAct;
    const core::PlayerId opponent = player.Other();
    result.state.playerToAct = opponent;

    if (action.Kind() == BettingActionKind::Fold)
    {
        result.transition = ActionTransition::Fold;
        return result;
    }

    if (action.Kind() == BettingActionKind::Check)
    {
        result.transition = state.lastActionWasCheck ? ActionTransition::BettingRoundComplete : ActionTransition::Continue;
        result.state.lastActionWasCheck = true;
    }
    else
    {
        const core::Chips chipsAdded = ChipsCommitted(state, action);
        result.state.pot += chipsAdded;
        result.state.stacks[player.Index()] -= chipsAdded;
        result.state.streetContributions[player.Index()] = action.AmountTo();
        result.state.lastActionWasCheck = false;

        if (action.Kind() == BettingActionKind::Call)
        {
            result.transition = ActionTransition::BettingRoundComplete;
        }
        else
        {
            const core::Chips raiseSize = action.AmountTo() - state.Contribution(opponent);
            if (action.Kind() == BettingActionKind::Bet || raiseSize >= state.lastFullRaiseSize)
                result.state.lastFullRaiseSize = raiseSize;
        }
    }

    if (TotalChips(result.state) != TotalChips(state))
        throw std::logic_error("Betting action did not conserve chips");
    return result;
}

core::Chips ChipsCommitted(const PublicState& state, const BettingAction& action)
{
    if (action.Kind() == BettingActionKind::Fold || action.Kind() == BettingActionKind::Check)
        return core::Chips{};

    const core::Chips chipsAdded = action.AmountTo() - state.Contribution(state.playerToAct);
    if (chipsAdded < core::Chips{})
        throw std::invalid_argument("Betting action amount-to is below the current contribution");
    return chipsAdded;
}
} // namespace solver::game
