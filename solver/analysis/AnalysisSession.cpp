#include "analysis/AnalysisSession.h"
#include "engine/StrategyEvaluator.h"
#include <algorithm>
#include "game/BettingRules.h"
#include "game/CompiledGame.h"
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

    if (node.Kind() == game::NodeKind::Terminal)
    {
        report.terminal = node.Terminal();
        return report;
    }

    if (node.Kind() == game::NodeKind::Chance)
    {
        const bool hasJointReach = !reachCalculator_.ReachFor(nodeId).jointReachMasses.empty();
        for (std::size_t edgeIndex = 0; edgeIndex < node.ChanceOutcomeCount(); ++edgeIndex)
        {
            const game::ChanceOutcome& outcome = node.GetChanceOutcome(edgeIndex);
            if (reachCalculator_.ReachFor(outcome.NextNode()).jointReachMasses.empty() && hasJointReach)
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
        const game::BettingAction& action = node.GetBettingEdge(edgeIndex).Action();
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

std::vector<HandReport> AnalysisSession::BuildHandReports(
    game::NodeId nodeId,
    const core::Range& range,
    const ReachCalculator::HandWeights& marginalReachMasses,
    const ReachCalculator::HandWeights& ownReachWeights,
    core::PlayerId player
) const
{
    std::map<core::HoleCards, float> evs;
    if (std::any_of(marginalReachMasses.begin(), marginalReachMasses.end(), [](const auto& entry) { return entry.second > 0.0f; }))
        evs = engine::EvaluateNodeStrategyEvs(result_.Problem(), result_.Strategy(), nodeId, player);
    std::vector<HandReport> hands;
    const core::Board& board = result_.Problem().game->GetNode(nodeId).State().board;
    for (const auto& [hand, inputRangeWeight] : range.Entries())
    {
        if (inputRangeWeight <= 0.0f || core::Overlaps(hand, board))
            continue;

        const auto marginalReachIt = marginalReachMasses.find(hand);
        const float marginalReachMass = marginalReachIt == marginalReachMasses.end() ? 0.0f : marginalReachIt->second;
        const auto ownReachIt = ownReachWeights.find(hand);
        const float ownReachWeight = ownReachIt == ownReachWeights.end() ? 0.0f : ownReachIt->second;
        std::optional<float> nodeStrategyEv;
        std::vector<float> strategy;
        if (marginalReachMass > 0.0f)
        {
            strategy = result_.Strategy().StrategyOrUniform({nodeId, hand});
            nodeStrategyEv = evs.at(hand);
        }

        hands.push_back({
            hand,
            inputRangeWeight,
            ownReachWeight,
            marginalReachMass,
            nodeStrategyEv,
            std::move(strategy),
        });
    }
    return hands;
}
} // namespace solver::analysis
