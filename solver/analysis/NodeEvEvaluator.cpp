#include "analysis/NodeEvEvaluator.h"
#include "game/CompiledGame.h"
#include "game/TerminalSettlement.h"
#include <stdexcept>

namespace solver::analysis
{
namespace
{
float TraverseNodeHandEv(
    const engine::SolveResult& result,
    game::NodeId nodeId,
    game::NodeId evRootNode,
    core::HoleCards player0Hand,
    core::HoleCards player1Hand,
    core::PlayerId player
)
{
    const game::CompiledGame& game = *result.Problem().game;
    const game::GameNode& node = game.GetNode(nodeId);
    if (node.Kind() == game::NodeKind::Terminal)
        return game::CalculateTerminalSettlement(game.GetNode(evRootNode), node, player0Hand, player1Hand).NetPayoffFromStart(player);

    if (node.Kind() == game::NodeKind::Chance)
    {
        float value = 0.0f;
        int legalOutcomeCount = 0;
        for (std::size_t edgeIndex = 0; edgeIndex < node.ChanceOutcomeCount(); ++edgeIndex)
        {
            const game::ChanceOutcome& outcome = node.GetChanceOutcome(edgeIndex);
            if (core::Contains(player0Hand, outcome.DealtCard()) || core::Contains(player1Hand, outcome.DealtCard()))
                continue;

            value += TraverseNodeHandEv(result, outcome.NextNode(), evRootNode, player0Hand, player1Hand, player);
            ++legalOutcomeCount;
        }
        if (legalOutcomeCount == 0)
            throw std::runtime_error("Chance node has no legal outcomes");
        return value / static_cast<float>(legalOutcomeCount);
    }

    const core::HoleCards actingHand = node.State().playerToAct == core::PlayerId::Player0() ? player0Hand : player1Hand;
    const float* strategy = result.Strategy().FindStrategy({nodeId, actingHand});
    const float uniformProbability = 1.0f / static_cast<float>(node.BettingEdgeCount());
    float value = 0.0f;
    for (std::size_t childIndex = 0; childIndex < node.BettingEdgeCount(); ++childIndex)
    {
        const float probability = strategy ? strategy[childIndex] : uniformProbability;
        if (probability == 0.0f)
            continue;

        value += probability *
                 TraverseNodeHandEv(result, node.GetBettingEdge(childIndex).NextNode(), evRootNode, player0Hand, player1Hand, player);
    }
    return value;
}

} // namespace

float CalculateNodeHandEv(
    const engine::SolveResult& result,
    game::NodeId nodeId,
    core::HoleCards hand,
    core::PlayerId player,
    const std::map<core::HoleCards, float>& opponentReachMasses
)
{
    const core::Board& board = result.Problem().game->GetNode(nodeId).State().board;
    if (core::Overlaps(hand, board))
        throw std::invalid_argument("Hand overlaps the public board");

    float totalValue = 0.0f;
    float totalOpponentWeight = 0.0f;
    for (const auto& [opponentHand, opponentReachMass] : opponentReachMasses)
    {
        if (opponentReachMass <= 0.0f || core::Overlaps(hand, opponentHand) || core::Overlaps(opponentHand, board))
            continue;

        const core::HoleCards player0Hand = player == core::PlayerId::Player0() ? hand : opponentHand;
        const core::HoleCards player1Hand = player == core::PlayerId::Player0() ? opponentHand : hand;
        totalValue += opponentReachMass * TraverseNodeHandEv(result, nodeId, nodeId, player0Hand, player1Hand, player);
        totalOpponentWeight += opponentReachMass;
    }
    if (totalOpponentWeight <= 0.0f)
        throw std::runtime_error("No valid opponent hands for node EV calculation");
    return totalValue / totalOpponentWeight;
}
} // namespace solver::analysis
