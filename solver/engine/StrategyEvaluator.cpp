#include "engine/StrategyEvaluator.h"
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
        // At the game root, opponent reach is the range weight and divisors are compatible opponent masses.
        std::vector<float> reach, divisors;
        for (const auto& hand : traversal.hands[1 - player])
            reach.push_back(hand.weight);
        for (const auto& hand : traversal.hands[player])
            divisors.push_back(hand.opponentMass);
        return strategy ? traversal.EvaluateSnapshot(*strategy, player, reach, divisors, HandTraversal::Evaluation::BestResponse)
                        : traversal.EvaluateAverageBestResponse(strategySums, player, reach, divisors);
    };
    return RootExploitability(*traversal.Data().tables, {values(0), values(1)});
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

NodeStrategyEvaluator::NodeStrategyEvaluator(const SolveResult& result) : problem_(result.Problem()), strategy_(result.Strategy()) {}

std::map<core::HoleCards, float> NodeStrategyEvaluator::Evaluate(game::NodeId node, core::PlayerId player)
{
    const auto board = problem_.game->GetNode(node).State().board;
    auto& tables = boards_[board.CardCount() - 3];
    if (!tables || tables->board != board)
    {
        // Release the previous board before constructing its replacement.
        tables.reset();
        tables = std::make_shared<const HandBoardData>(problem_, node);
    }
    const HandTraversal traversal(std::make_shared<const HandTraversalData>(tables, node));
    const auto reach = traversal.OpponentReachAtRoot(strategy_, player.Other().Index());
    const auto divisors = traversal.CompatibleMasses(player.Index(), reach.data());
    const auto values = traversal.EvaluateSnapshot(strategy_, player.Index(), reach, divisors, HandTraversal::Evaluation::StrategyValue);
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
