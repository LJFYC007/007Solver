#include "analysis/AnalysisSession.h"
#include "engine/StrategyEvaluator.h"
#include <algorithm>
#include "game/BettingRules.h"
#include "game/CompiledGame.h"
#include <cstdint>
#include <utility>

namespace solver::analysis
{
AnalysisSession::AnalysisSession(engine::SolveResult result) : result_(std::move(result)), reachCalculator_(result_) {}

NodeReport AnalysisSession::QueryNode(game::NodeId nodeId)
{
    const game::GameNode& node = result_.Problem().game->GetNode(nodeId);
    NodeReport report{
        nodeId,
        node.Kind(),
        {node.State().street, node.State().board, node.State().pot, node.State().stacks},
    };

    const auto& currentReach = reachCalculator_.ReachFor(nodeId);
    for (const auto& [hand, weight] : currentReach.ownReachWeights.player0)
        if (!core::Overlaps(hand, node.State().board))
            report.state.rangeCombos[0] += weight;
    for (const auto& [hand, weight] : currentReach.ownReachWeights.player1)
        if (!core::Overlaps(hand, node.State().board))
            report.state.rangeCombos[1] += weight;

    if (node.Kind() == game::NodeKind::Terminal)
    {
        report.terminal = node.Terminal();
        return report;
    }

    if (node.Kind() == game::NodeKind::Chance)
    {
        // A card is unavailable only if every supported private pair blocks it.
        // Inspect the current reach without populating caches for unvisited children.
        std::uint64_t blockedByAll = (std::uint64_t{1} << 52) - 1;
        bool hasJointReach = false;
        for (const auto& jointReach : currentReach.jointReachMasses)
        {
            if (jointReach.jointReachMass <= 0.0f)
                continue;
            hasJointReach = true;
            std::uint64_t blocked = 0;
            for (core::Card card : jointReach.player0Hand.Cards())
                blocked |= std::uint64_t{1} << card.Index();
            for (core::Card card : jointReach.player1Hand.Cards())
                blocked |= std::uint64_t{1} << card.Index();
            blockedByAll &= blocked;
            if (blockedByAll == 0)
                break;
        }
        for (std::size_t edgeIndex = 0; edgeIndex < node.ChanceOutcomeCount(); ++edgeIndex)
        {
            const game::ChanceOutcome& outcome = node.GetChanceOutcome(edgeIndex);
            if (hasJointReach && (blockedByAll & (std::uint64_t{1} << outcome.DealtCard().Index())) != 0)
                continue;

            report.outcomes.push_back({outcome.DealtCard(), outcome.NextNode()});
        }
        return report;
    }

    const core::PlayerId player = node.State().playerToAct;
    report.actor = player;
    const ReachCalculator::NodeReach& reach = reachCalculator_.ReachFor(nodeId);
    const ReachCalculator::HandWeights marginalReachMasses = reachCalculator_.BuildMarginalReachMasses(reach.jointReachMasses, player);
    report.hands = BuildHandReports(
        nodeId,
        result_.Problem().ranges.For(player),
        marginalReachMasses,
        player == core::PlayerId::Player0() ? reach.ownReachWeights.player0 : reach.ownReachWeights.player1,
        player
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
        const auto evs = engine::EvaluateNodeStrategyEvs(result_.Problem(), result_.Strategy(), report.nodeId, *report.actor);
        for (auto& hand : report.hands)
            if (hand.marginalReachMass > 0.0f)
                hand.nodeStrategyEv = evs.at(hand.cards);
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
