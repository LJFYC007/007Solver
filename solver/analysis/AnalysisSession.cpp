#include "analysis/AnalysisSession.h"
#include "engine/StrategyEvaluator.h"
#include <algorithm>
#include "game/BettingRules.h"
#include "game/CompiledGame.h"
#include <cstdint>
#include <utility>

namespace solver::analysis
{
AnalysisSession::AnalysisSession(engine::SolveResult result)
    : result_(std::move(result)), reachCalculator_(result_), nodeEvaluator_(result_)
{}

NodeReport AnalysisSession::QueryNode(game::NodeId nodeId)
{
    const game::GameNode& node = result_.Problem().game->GetNode(nodeId);
    NodeReport report{
        nodeId,
        node.Kind(),
        {node.State().street, node.State().board, node.State().pot, node.State().stacks},
    };

    const auto& currentReach = reachCalculator_.ReachFor(nodeId);
    for (std::size_t player = 0; player < 2; ++player)
        for (const auto& [hand, weight] : currentReach.ownReachWeights[player])
            if (!core::Overlaps(hand, node.State().board))
                report.state.rangeCombos[player] += weight;

    if (node.Kind() == game::NodeKind::Terminal)
    {
        report.terminal = node.Terminal();
        return report;
    }

    if (node.Kind() == game::NodeKind::Chance)
    {
        // A card is unavailable only if every supported private pair blocks it.
        // Inspect the current reach without populating caches for unvisited children.
        const auto blockedByAll = reachCalculator_.CommonBlockers(currentReach, node.State().board);
        for (std::size_t edgeIndex = 0; edgeIndex < node.ChanceOutcomeCount(); ++edgeIndex)
        {
            const game::ChanceOutcome& outcome = node.GetChanceOutcome(edgeIndex);
            if (blockedByAll && (*blockedByAll & (std::uint64_t{1} << outcome.DealtCard().Index())) != 0)
                continue;

            report.outcomes.push_back({outcome.DealtCard(), outcome.NextNode()});
        }
        return report;
    }

    const core::PlayerId player = node.State().playerToAct;
    report.actor = player;
    const ReachCalculator::HandWeights marginalReachMasses =
        reachCalculator_.BuildMarginalReachMasses(currentReach, node.State().board, player);
    report.hands = BuildHandReports(
        nodeId, result_.Problem().ranges.For(player), marginalReachMasses, currentReach.ownReachWeights[player.Index()], player
    );

    for (std::size_t edgeIndex = 0; edgeIndex < node.BettingEdgeCount(); ++edgeIndex)
    {
        const game::BettingAction action = node.GetBettingEdge(edgeIndex).Action();
        report.actions.push_back({
            action.Kind(),
            game::IsAllIn(node.State(), action),
            action.AmountTo(),
            game::ChipsCommitted(node.State(), action),
            node.GetBettingEdge(edgeIndex).NextNode(),
        });
    }
    return report;
}

NodeReport AnalysisSession::EvaluateNodeEvs(NodeReport report) const
{
    if (report.kind != game::NodeKind::Decision)
        return report;
    if (std::any_of(report.hands.begin(), report.hands.end(), [](const auto& hand) { return hand.marginalReachMass > 0.0f; }))
    {
        const auto evs = nodeEvaluator_.Evaluate(report.nodeId, *report.actor);
        for (auto& hand : report.hands)
        {
            // The evaluator omits hands whose opponent mass has no value scale (see ValueScale).
            const auto ev = evs.find(hand.cards);
            if (hand.marginalReachMass > 0.0f && ev != evs.end())
                hand.nodeStrategyEv = ev->second;
        }
    }
    report.evsReady = true;
    return report;
}

std::vector<HandReport> AnalysisSession::BuildHandReports(
    game::NodeId nodeId,
    const core::Range& range,
    const ReachCalculator::HandWeights& marginalReachMasses,
    const ReachCalculator::HandWeights& ownReachWeights,
    core::PlayerId player
) const
{
    std::vector<HandReport> hands;
    const core::Board board = result_.Problem().game->GetNode(nodeId).State().board;
    for (const auto& [hand, inputRangeWeight] : range.Entries())
    {
        if (inputRangeWeight <= 0.0f || core::Overlaps(hand, board))
            continue;

        const auto marginalReachIt = marginalReachMasses.find(hand);
        const float marginalReachMass = marginalReachIt == marginalReachMasses.end() ? 0.0f : marginalReachIt->second;
        const auto ownReachIt = ownReachWeights.find(hand);
        const float ownReachWeight = ownReachIt == ownReachWeights.end() ? 0.0f : ownReachIt->second;
        std::vector<float> strategy;
        if (marginalReachMass > 0.0f)
            strategy = result_.Strategy().StrategyOrUniform({nodeId, hand});

        hands.push_back({
            hand,
            inputRangeWeight,
            ownReachWeight,
            marginalReachMass,
            std::nullopt,
            std::move(strategy),
        });
    }
    return hands;
}
} // namespace solver::analysis
