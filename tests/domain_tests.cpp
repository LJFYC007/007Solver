#include "core/Card.h"
#include "core/Chips.h"
#include "game/BettingAbstraction.h"
#include "game/BettingRules.h"
#include <cmath>
#include <vector>
#include <gtest/gtest.h>

namespace
{
namespace core = solver::core;
namespace game = solver::game;

core::Chips Chips(double value)
{
    return core::Chips::FromRaw(static_cast<std::int32_t>(std::lround(value * core::Chips::kUnitsPerChip)));
}

game::PublicState MakeState(double pot = 10.0, double player0Stack = 100.0, double player1Stack = 100.0)
{
    return {
        core::PlayerId::Player0(),
        Chips(pot),
        {Chips(player0Stack), Chips(player1Stack)},
        {Chips(0.0), Chips(0.0)},
        core::Street::Flop,
        core::ParseBoard("2c 3d 4h", 3),
    };
}

} // namespace

TEST(BettingRulesTest, FullRaiseAndShortAllInRaiseHaveDistinctRaiseSizeEffects)
{
    const game::ActionApplication afterBet = game::ApplyAction(MakeState(), game::BettingAction(game::BettingActionKind::Bet, Chips(10.0)));
    const game::ActionApplication afterFullRaise =
        game::ApplyAction(afterBet.state, game::BettingAction(game::BettingActionKind::Raise, Chips(30.0)));
    EXPECT_EQ(afterFullRaise.state.lastFullRaiseSize, Chips(20.0));
    EXPECT_EQ(game::AmountToCall(afterFullRaise.state), Chips(20.0));
    EXPECT_EQ(game::ChipsCommitted(afterFullRaise.state, game::BettingAction(game::BettingActionKind::Call, Chips(30.0))), Chips(20.0));

    const game::PublicState shortStackRoot = MakeState(10.0, 100.0, 15.0);
    const game::ActionApplication beforeShortRaise =
        game::ApplyAction(shortStackRoot, game::BettingAction(game::BettingActionKind::Bet, Chips(10.0)));
    const game::BettingAction shortRaise(game::BettingActionKind::Raise, Chips(15.0));
    ASSERT_TRUE(game::IsAllIn(beforeShortRaise.state, shortRaise));
    const game::ActionApplication afterShortRaise = game::ApplyAction(beforeShortRaise.state, shortRaise);
    EXPECT_EQ(afterShortRaise.state.lastFullRaiseSize, Chips(10.0));
    EXPECT_EQ(game::AmountToCall(afterShortRaise.state), Chips(5.0));
}

TEST(BettingRulesTest, UnequalStacksCapMaximumWithoutMakingItAllIn)
{
    const game::PublicState state = MakeState(10.0, 100.0, 40.0);
    const game::LegalActionSet legalActions = game::GetLegalActions(state);
    ASSERT_TRUE(legalActions.aggression.has_value());
    EXPECT_EQ(legalActions.aggression->maximumAmountTo, Chips(40.0));

    const std::vector<game::BettingAction> actions = game::BettingAbstraction::Default().SelectActions(state, legalActions);
    const game::BettingAction& maximum = actions.back();
    EXPECT_EQ(maximum.AmountTo(), Chips(40.0));
    EXPECT_FALSE(game::IsAllIn(state, maximum));
}

TEST(BettingAbstractionTest, BetAndRaiseSizesUseExplicitAmountToAndPotBases)
{
    const game::PublicState root = MakeState();
    const game::LegalActionSet rootActions = game::GetLegalActions(root);
    game::BettingAbstraction abstraction;
    abstraction.betSizes = {
        {game::BetSizeKind::PotFractionOfCurrentPot, 1, 2, Chips(0.0)},
        {game::BetSizeKind::AbsoluteAmountTo, 0, 1, Chips(7.0)},
    };
    abstraction.includeMaximumBet = false;
    const std::vector<game::BettingAction> bets = abstraction.SelectActions(root, rootActions);
    ASSERT_EQ(bets.size(), 3u);
    EXPECT_EQ(bets[1].AmountTo(), Chips(5.0));
    EXPECT_EQ(bets[2].AmountTo(), Chips(7.0));

    const game::ActionApplication afterBet = game::ApplyAction(root, game::BettingAction(game::BettingActionKind::Bet, Chips(10.0)));
    const game::LegalActionSet facingActions = game::GetLegalActions(afterBet.state);
    abstraction.betSizes.clear();
    abstraction.raiseSizes = {
        {game::RaiseSizeKind::PotFractionOfPotAfterCallAsRaiseBy, 1, 2, Chips(0.0)},
        {game::RaiseSizeKind::AbsoluteAmountTo, 0, 1, Chips(30.0)},
    };
    abstraction.includeMaximumRaise = false;
    const std::vector<game::BettingAction> raises = abstraction.SelectActions(afterBet.state, facingActions);
    ASSERT_EQ(raises.size(), 4u);
    EXPECT_EQ(raises[2].AmountTo(), Chips(25.0));
    EXPECT_EQ(raises[3].AmountTo(), Chips(30.0));
}
