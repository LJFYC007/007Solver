#include "engine/StrategyEvaluator.h"
#include "engine/HandTraversal.h"
#include <stdexcept>
#include <vector>

namespace solver::engine
{
namespace
{
void CheckProblem(const SolveProblem& problem, const StrategySnapshot& strategy)
{
    if (!problem.game || problem.game.get() != &strategy.Game())
        throw std::invalid_argument("Evaluation strategy belongs to a different game");
}

ExploitabilityMetrics EvaluateBestResponses(const HandTraversal& traversal, const StrategySnapshot* strategy, const float* strategySums)
{
    std::array<float, 2> bestResponses{};
    for (std::size_t player = 0; player < 2; ++player)
    {
        std::vector<float> reach;
        if (strategy)
            reach = traversal.OpponentReachAtRoot(*strategy, 1 - player);
        else
            for (const auto& hand : traversal.hands[1 - player])
                reach.push_back(hand.weight);
        const auto divisors = traversal.CompatibleMasses(player, reach.data());
        const auto values = strategy
                                ? traversal.EvaluateSnapshot(*strategy, player, reach, divisors, HandTraversal::Evaluation::BestResponse)
                                : traversal.EvaluateAverageBestResponse(strategySums, player, reach, divisors);
        float totalValue = 0.0f, totalMass = 0.0f;
        for (std::size_t hand = 0; hand < values.size(); ++hand)
        {
            const float mass = traversal.hands[player][hand].weight * divisors[hand];
            totalValue += mass * values[hand];
            totalMass += mass;
        }
        bestResponses[player] = totalValue / totalMass;
    }
    return {bestResponses[0], bestResponses[1], (bestResponses[0] + bestResponses[1]) / 2.0f};
}
} // namespace

ExploitabilityMetrics EvaluateExploitability(const SolveProblem& problem, const StrategySnapshot& strategy)
{
    CheckProblem(problem, strategy);
    return EvaluateBestResponses(HandTraversal(problem, problem.game->Root()), &strategy, nullptr);
}

ExploitabilityMetrics EvaluateAverageStrategy(const HandTraversal& traversal, const float* strategySums)
{
    return EvaluateBestResponses(traversal, nullptr, strategySums);
}

std::map<core::HoleCards, float> EvaluateNodeStrategyEvs(
    const SolveProblem& problem,
    const StrategySnapshot& strategy,
    game::NodeId node,
    core::PlayerId player
)
{
    CheckProblem(problem, strategy);
    const HandTraversal traversal(problem, node);
    const auto reach = traversal.OpponentReachAtRoot(strategy, player.Other().Index());
    const auto divisors = traversal.CompatibleMasses(player.Index(), reach.data());
    const auto values = traversal.EvaluateSnapshot(strategy, player.Index(), reach, divisors, HandTraversal::Evaluation::StrategyValue);
    std::map<core::HoleCards, float> evs;
    for (std::size_t hand = 0; hand < values.size(); ++hand)
    {
        if (divisors[hand] > 0.0f)
        {
            // Undo the subtree's zero-sum shift for either player: the pot at the
            // queried node is dead money; this restores net payoff from that node.
            evs.emplace(traversal.hands[player.Index()][hand].cards, values[hand] + traversal.rootHalfPot);
        }
    }
    return evs;
}
} // namespace solver::engine
