#include "engine/StrategyEvaluator.h"
#include "engine/HandEvaluation.h"
#include "engine/HandTraversal.h"
#include <stdexcept>
#include <vector>

namespace solver::engine
{
namespace
{
ExploitabilityMetrics EvaluateRootExploitability(
    const HandTraversal& traversal,
    const StrategySnapshot* strategy,
    const std::uint16_t* strategySums
)
{
    return RootExploitability(
        traversal.Data().tables,
        [&](std::size_t player, bool policy)
        {
            // At the game root, opponent reach is the range weight and values scale by the compatible
            // opponent masses.
            std::vector<float> reach, scales;
            for (const auto& hand : traversal.hands[1 - player])
                reach.push_back(hand.weight);
            for (const auto& hand : traversal.hands[player])
                scales.push_back(ValueScale(hand.opponentMass));
            // One walk shares reach and payoffs between both value rows.
            std::vector<HandTraversal::Evaluation> evaluations{HandTraversal::Evaluation::BestResponse};
            if (policy)
                evaluations.push_back(HandTraversal::Evaluation::StrategyValue);
            return strategy ? traversal.EvaluateSnapshot(*strategy, player, reach, scales, evaluations)
                            : traversal.EvaluateAverage(strategySums, player, reach, scales, evaluations);
        }
    );
}
} // namespace

ExploitabilityMetrics RootExploitability(const HandBoardData& tables, const RootValues& rootValues)
{
    // Raked policy utilities do not sum to zero, so raked games also evaluate them.
    const bool raked = tables.game->Spec().HasRake();
    // Double sums, as in the independent reference: float sums of wide ranges drift by about 1e-6.
    std::array<double, 2> bestResponses{};
    std::array<double, 2> policyEvs{};
    for (std::size_t player = 0; player < 2; ++player)
    {
        const auto values = rootValues(player, raked);
        const std::size_t count = tables.hands[player].size();
        double totalValue = 0.0, totalPolicyValue = 0.0, totalMass = 0.0;
        for (std::size_t hand = 0; hand < count; ++hand)
        {
            const auto& source = tables.hands[player][hand];
            const double mass = static_cast<double>(source.weight) * source.opponentMass;
            totalValue += mass * values[hand];
            if (raked)
                totalPolicyValue += mass * values[count + hand];
            totalMass += mass;
        }
        bestResponses[player] = totalValue / totalMass;
        policyEvs[player] = totalPolicyValue / totalMass;
    }
    return {
        static_cast<float>(bestResponses[0]),
        static_cast<float>(bestResponses[1]),
        static_cast<float>((bestResponses[0] - policyEvs[0] + bestResponses[1] - policyEvs[1]) / 2.0),
    };
}

ExploitabilityMetrics EvaluateExploitability(const SolveProblem& problem, const StrategySnapshot& strategy)
{
    if (!problem.game || problem.game.get() != &strategy.Game())
        throw std::invalid_argument("Evaluation strategy belongs to a different game");
    return EvaluateRootExploitability(HandTraversal(problem, problem.game->Root()), &strategy, nullptr);
}

ExploitabilityMetrics EvaluateAverageStrategy(const HandTraversal& traversal, const std::uint16_t* strategySums)
{
    return EvaluateRootExploitability(traversal, nullptr, strategySums);
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
    const auto& root = traversal.nodes.front();
    const bool rake =
        result.Problem().game->Spec().HasRake() && root.kind == HandTraversalData::Kind::Decision && root.actor == player.Index();
    std::vector<HandTraversal::Evaluation> evaluations{HandTraversal::Evaluation::StrategyValue};
    if (rake)
        evaluations.push_back(HandTraversal::Evaluation::ExpectedRake);
    std::vector<float> actionValues;
    const auto values = traversal.EvaluateSnapshot(strategy, player.Index(), reach, scales, evaluations, &actionValues);
    const std::size_t count = scales.size();
    std::map<core::HoleCards, NodeStrategyValue> evs;
    for (std::size_t hand = 0; hand < count; ++hand)
    {
        if (scales[hand] <= 0.0f)
            continue;
        // Undo the subtree's half-pot shift for either player: the pot at the queried node is
        // dead money; this restores net payoff from that node. Action values share the frame
        // and include the action's own contribution.
        NodeStrategyValue value{values[hand] + traversal.rootHalfPot, {}, rake ? values[count + hand] : 0.0f};
        for (std::size_t action = 0; action < actionValues.size() / count; ++action)
            value.actionEvs.push_back(actionValues[action * count + hand] + traversal.rootHalfPot);
        evs.emplace(traversal.hands[player.Index()][hand].cards, std::move(value));
    }
    return evs;
}
} // namespace solver::engine
