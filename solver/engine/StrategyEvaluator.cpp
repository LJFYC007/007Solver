#include "engine/StrategyEvaluator.h"
#include "engine/HandTraversal.h"
#include <algorithm>
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

// A fixed snapshot needs opponent reach only, and no regrets or average-strategy buffers.
std::vector<float> EvaluateHands(
    const HandTraversal& traversal,
    const StrategySnapshot& strategy,
    std::size_t player,
    const std::vector<double>& rootReach,
    const std::vector<double>& divisors,
    bool bestResponse
)
{
    const std::size_t count = traversal.hands[player].size();
    auto workspace = traversal.MakeWorkspace();
    std::copy(rootReach.begin(), rootReach.end(), workspace.reach[1 - player].begin());
    std::vector<float> values(count);
    traversal.Walk(0, player, &strategy, divisors.data(), bestResponse, workspace, 0, values.data());
    return values;
}
} // namespace

ExploitabilityMetrics EvaluateExploitability(const SolveProblem& problem, const StrategySnapshot& strategy)
{
    CheckProblem(problem, strategy);
    const HandTraversal traversal(problem, problem.game->Root());
    std::array<float, 2> bestResponses{};
    for (std::size_t player = 0; player < 2; ++player)
    {
        const auto reach = traversal.OpponentReachAtRoot(strategy, 1 - player);
        const auto divisors = traversal.CompatibleMasses(player, reach.data());
        const auto values = EvaluateHands(traversal, strategy, player, reach, divisors, true);
        double totalValue = 0.0;
        double totalMass = 0.0;
        for (std::size_t hand = 0; hand < values.size(); ++hand)
        {
            const double mass = traversal.hands[player][hand].weight * divisors[hand];
            totalValue += mass * values[hand];
            totalMass += mass;
        }
        bestResponses[player] = static_cast<float>(totalValue / totalMass);
    }
    return {bestResponses[0], bestResponses[1], (bestResponses[0] + bestResponses[1]) / 2.0f};
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
    const auto values = EvaluateHands(traversal, strategy, player.Index(), reach, divisors, false);
    std::map<core::HoleCards, float> evs;
    for (std::size_t hand = 0; hand < values.size(); ++hand)
    {
        if (divisors[hand] > 0.0)
        {
            // Undo the subtree's zero-sum shift for either player: the pot at the
            // queried node is dead money; this restores net payoff from that node.
            evs.emplace(traversal.hands[player.Index()][hand].cards, values[hand] + traversal.rootHalfPot);
        }
    }
    return evs;
}
} // namespace solver::engine
