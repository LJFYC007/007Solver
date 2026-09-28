#include "engine/StrategyEvaluator.h"
#include "engine/HandEvaluation.h"
#include "engine/HandTraversal.h"
#include <stdexcept>
#include <vector>

namespace solver::engine
{
namespace
{
ExploitabilityMetrics EvaluateBestResponses(
    const HandTraversal& traversal,
    const StrategySnapshot* strategy,
    const std::uint16_t* strategySums
)
{
    const auto values = [&](std::size_t player)
    {
        // At the game root, opponent reach is the range weight and values scale by the compatible
        // opponent masses.
        std::vector<float> reach, scales;
        for (const auto& hand : traversal.hands[1 - player])
            reach.push_back(hand.weight);
        for (const auto& hand : traversal.hands[player])
            scales.push_back(ValueScale(hand.opponentMass));
        return strategy ? traversal.EvaluateSnapshot(*strategy, player, reach, scales, HandTraversal::Evaluation::BestResponse)
                        : traversal.EvaluateAverageBestResponse(strategySums, player, reach, scales);
    };
    return RootExploitability(traversal.Data().tables, {values(0), values(1)});
}
} // namespace

ExploitabilityMetrics RootExploitability(const HandBoardData& tables, const std::array<std::vector<float>, 2>& bestResponseValues)
{
    std::array<float, 2> bestResponses{};
    for (std::size_t player = 0; player < 2; ++player)
    {
        const auto& values = bestResponseValues[player];
        float totalValue = 0.0f, totalMass = 0.0f;
        for (std::size_t hand = 0; hand < values.size(); ++hand)
        {
            const auto& source = tables.hands[player][hand];
            const float mass = source.weight * source.opponentMass;
            totalValue += mass * values[hand];
            totalMass += mass;
        }
        bestResponses[player] = totalValue / totalMass;
    }
    return {bestResponses[0], bestResponses[1], (bestResponses[0] + bestResponses[1]) / 2.0f};
}

ExploitabilityMetrics EvaluateExploitability(const SolveProblem& problem, const StrategySnapshot& strategy)
{
    if (!problem.game || problem.game.get() != &strategy.Game())
        throw std::invalid_argument("Evaluation strategy belongs to a different game");
    return EvaluateBestResponses(HandTraversal(problem, problem.game->Root()), &strategy, nullptr);
}

ExploitabilityMetrics EvaluateAverageStrategy(const HandTraversal& traversal, const std::uint16_t* strategySums)
{
    return EvaluateBestResponses(traversal, nullptr, strategySums);
}

std::map<core::HoleCards, NodeStrategyValue> EvaluateNodeStrategy(
    const SolveResult& result,
    const HandTraversal& traversal,
    core::PlayerId player
)
{
    const auto& strategy = result.Strategy();
    const auto reach = traversal.OpponentReachAtRoot(strategy, player.Other().Index());
    const auto scales = ValueScales(traversal.CompatibleMasses(player.Index(), reach.data()));
    std::vector<float> actionValues;
    const auto values =
        traversal.EvaluateSnapshot(strategy, player.Index(), reach, scales, HandTraversal::Evaluation::StrategyValue, &actionValues);
    const std::size_t count = values.size();
    std::map<core::HoleCards, NodeStrategyValue> evs;
    for (std::size_t hand = 0; hand < count; ++hand)
    {
        if (scales[hand] <= 0.0f)
            continue;
        // Undo the subtree's zero-sum shift for either player: the pot at the queried node is
        // dead money; this restores net payoff from that node. Action values share the frame
        // and include the action's own contribution.
        NodeStrategyValue value{values[hand] + traversal.rootHalfPot, {}};
        for (std::size_t action = 0; action < actionValues.size() / count; ++action)
            value.actionEvs.push_back(actionValues[action * count + hand] + traversal.rootHalfPot);
        evs.emplace(traversal.hands[player.Index()][hand].cards, std::move(value));
    }
    return evs;
}
} // namespace solver::engine
