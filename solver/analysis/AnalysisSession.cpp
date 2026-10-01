#include "analysis/AnalysisSession.h"
#include "engine/HandTraversal.h"
#include "engine/StrategyEvaluator.h"
#include <algorithm>
#include "game/BettingRules.h"
#include "game/CompiledGame.h"
#include "game/TerminalSettlement.h"
#include <cstdint>
#include <map>
#include <stdexcept>
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
    for (std::size_t player = 0; player < 2; ++player)
        for (const auto& [hand, weight] : currentReach.ownReachWeights[player])
            if (!core::Overlaps(hand, node.State().board))
                report.state.rangeCombos[player] += weight;

    if (node.Kind() == game::NodeKind::Terminal)
    {
        report.terminal = node.Terminal();
        report.rake = game::TerminalRake(result_.Problem().game->Spec(), node.State());
        return report;
    }

    if (node.Kind() == game::NodeKind::Chance)
    {
        if (node.IsForcedRunout())
            report.rake = game::TerminalRake(result_.Problem().game->Spec(), node.State());
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
    const auto evs = JointReachEvs(nodeId, player, marginalReachMasses);
    // Joint reach weights the hands with EVs and the rake they expect to pay.
    double value = 0.0, rake = 0.0, mass = 0.0;
    for (auto& hand : report.hands)
    {
        const auto ev = evs.find(hand.cards);
        if (ev != evs.end())
        {
            hand.nodeStrategyEv = ev->second.ev;
            hand.actionEvs = ev->second.actionEvs;
            value += static_cast<double>(hand.marginalReachMass) * ev->second.ev;
            rake += static_cast<double>(hand.marginalReachMass) * ev->second.expectedRake;
            mass += hand.marginalReachMass;
        }
    }
    if (mass > 0.0)
    {
        // Each terminal's two node EVs sum to the pot at this node less the rake it pays.
        report.rangeEvs[player.Index()] = static_cast<float>(value / mass);
        report.rangeEvs[player.Other().Index()] = static_cast<float>(core::ToChipUnits(node.State().pot) - (value + rake) / mass);
    }

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

OpponentEvReport AnalysisSession::QueryOpponentEv(game::NodeId nodeId)
{
    const game::GameNode& node = result_.Problem().game->GetNode(nodeId);
    if (node.Kind() != game::NodeKind::Decision)
        throw std::invalid_argument("Opponent EVs need a decision node");
    const core::PlayerId player = node.State().playerToAct.Other();
    const core::Board board = node.State().board;
    const auto masses = reachCalculator_.BuildMarginalReachMasses(reachCalculator_.ReachFor(nodeId), board, player);
    const auto evs = JointReachEvs(nodeId, player, masses);
    OpponentEvReport report{nodeId, player, {}};
    for (const auto& [hand, inputRangeWeight] : result_.Problem().ranges.For(player).Entries())
    {
        if (inputRangeWeight <= 0.0f || core::Overlaps(hand, board))
            continue;
        const auto mass = masses.find(hand);
        OpponentHandReport entry{hand, mass == masses.end() ? 0.0f : mass->second, std::nullopt};
        const auto ev = evs.find(hand);
        if (ev != evs.end())
            entry.nodeStrategyEv = ev->second.ev;
        report.hands.push_back(entry);
    }
    return report;
}

engine::HandTraversal AnalysisSession::TraversalAt(game::NodeId nodeId)
{
    if (!traversal_ || traversal_->nodes.front().id != nodeId)
    {
        // Free the previous tables before building the next.
        traversal_.reset();
        traversal_ = std::make_shared<const engine::HandTraversalData>(result_.Problem(), nodeId);
    }
    return engine::HandTraversal(traversal_);
}

std::map<core::HoleCards, engine::NodeStrategyValue> AnalysisSession::JointReachEvs(
    game::NodeId nodeId,
    core::PlayerId player,
    const ReachCalculator::HandWeights& marginalReachMasses
)
{
    std::map<core::HoleCards, engine::NodeStrategyValue> evs;
    if (std::none_of(marginalReachMasses.begin(), marginalReachMasses.end(), [](const auto& entry) { return entry.second > 0.0f; }))
        return evs;
    evs = engine::EvaluateNodeStrategy(result_, TraversalAt(nodeId), player);
    for (auto ev = evs.begin(); ev != evs.end();)
    {
        const auto mass = marginalReachMasses.find(ev->first);
        ev = mass != marginalReachMasses.end() && mass->second > 0.0f ? std::next(ev) : evs.erase(ev);
    }
    return evs;
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
            {},
        });
    }
    return hands;
}
} // namespace solver::analysis
