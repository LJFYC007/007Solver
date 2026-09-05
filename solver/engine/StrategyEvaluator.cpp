#include "engine/StrategyEvaluator.h"
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace solver::engine
{
namespace
{
std::vector<float> TraverseBestResponse(
    const SolveProblem& problem,
    const StrategySnapshot& strategy,
    game::NodeId nodeId,
    core::HoleCards responderHand,
    const std::vector<core::HoleCards>& opponentHands,
    const std::vector<float>& opponentReach,
    core::PlayerId responder
)
{
    const game::GameNode& node = problem.game->GetNode(nodeId);
    std::vector<float> values(opponentHands.size(), 0.0f);

    if (node.Kind() == game::NodeKind::Terminal)
    {
        for (std::size_t handIndex = 0; handIndex < opponentHands.size(); ++handIndex)
        {
            if (opponentReach[handIndex] <= 0.0f)
                continue;

            const auto utility = problem.game->CalculateZeroSumUtility(
                nodeId,
                responder == core::PlayerId::Player0() ? responderHand : opponentHands[handIndex],
                responder == core::PlayerId::Player0() ? opponentHands[handIndex] : responderHand
            );
            values[handIndex] = responder == core::PlayerId::Player0() ? utility.first : utility.second;
        }
        return values;
    }

    if (node.Kind() == game::NodeKind::Chance)
    {
        std::vector<int> legalOutcomeCounts(opponentHands.size(), 0);
        for (std::size_t outcomeIndex = 0; outcomeIndex < node.ChanceOutcomeCount(); ++outcomeIndex)
        {
            const game::ChanceOutcome& outcome = node.GetChanceOutcome(outcomeIndex);
            for (std::size_t handIndex = 0; handIndex < opponentHands.size(); ++handIndex)
            {
                if (opponentReach[handIndex] > 0.0f && !core::Contains(responderHand, outcome.DealtCard()) &&
                    !core::Contains(opponentHands[handIndex], outcome.DealtCard()))
                    ++legalOutcomeCounts[handIndex];
            }
        }

        for (std::size_t outcomeIndex = 0; outcomeIndex < node.ChanceOutcomeCount(); ++outcomeIndex)
        {
            const game::ChanceOutcome& outcome = node.GetChanceOutcome(outcomeIndex);
            std::vector<float> childReach(opponentHands.size(), 0.0f);
            bool hasChildReach = false;
            for (std::size_t handIndex = 0; handIndex < opponentHands.size(); ++handIndex)
            {
                if (legalOutcomeCounts[handIndex] > 0 && !core::Contains(responderHand, outcome.DealtCard()) &&
                    !core::Contains(opponentHands[handIndex], outcome.DealtCard()))
                {
                    childReach[handIndex] = opponentReach[handIndex] / static_cast<float>(legalOutcomeCounts[handIndex]);
                    hasChildReach = hasChildReach || childReach[handIndex] > 0.0f;
                }
            }
            if (!hasChildReach)
                continue;

            const std::vector<float> childValues =
                TraverseBestResponse(problem, strategy, outcome.NextNode(), responderHand, opponentHands, childReach, responder);
            for (std::size_t handIndex = 0; handIndex < opponentHands.size(); ++handIndex)
            {
                if (legalOutcomeCounts[handIndex] > 0 && childReach[handIndex] > 0.0f)
                    values[handIndex] += childValues[handIndex] / static_cast<float>(legalOutcomeCounts[handIndex]);
            }
        }
        return values;
    }

    const bool isResponderNode = node.State().playerToAct == responder;
    if (isResponderNode)
    {
        float bestValue = -std::numeric_limits<float>::infinity();
        std::vector<float> bestActionValues;
        for (std::size_t actionIndex = 0; actionIndex < node.BettingEdgeCount(); ++actionIndex)
        {
            std::vector<float> childValues = TraverseBestResponse(
                problem, strategy, node.GetBettingEdge(actionIndex).NextNode(), responderHand, opponentHands, opponentReach, responder
            );
            float actionValue = 0.0f;
            for (std::size_t handIndex = 0; handIndex < opponentHands.size(); ++handIndex)
                actionValue += opponentReach[handIndex] * childValues[handIndex];

            if (actionValue > bestValue)
            {
                bestValue = actionValue;
                bestActionValues = std::move(childValues);
            }
        }
        return bestActionValues;
    }

    const float uniformProbability = 1.0f / static_cast<float>(node.BettingEdgeCount());
    std::vector<const float*> opponentStrategies(opponentHands.size(), nullptr);
    for (std::size_t handIndex = 0; handIndex < opponentHands.size(); ++handIndex)
    {
        if (opponentReach[handIndex] > 0.0f)
        {
            opponentStrategies[handIndex] = strategy.FindStrategy({nodeId, opponentHands[handIndex]});
        }
    }

    for (std::size_t actionIndex = 0; actionIndex < node.BettingEdgeCount(); ++actionIndex)
    {
        std::vector<float> childReach(opponentHands.size(), 0.0f);
        bool hasChildReach = false;
        for (std::size_t handIndex = 0; handIndex < opponentHands.size(); ++handIndex)
        {
            if (opponentReach[handIndex] > 0.0f)
            {
                const float* probabilities = opponentStrategies[handIndex];
                const float probability = probabilities ? probabilities[actionIndex] : uniformProbability;
                childReach[handIndex] = opponentReach[handIndex] * probability;
                hasChildReach = hasChildReach || childReach[handIndex] > 0.0f;
            }
        }
        if (!hasChildReach)
            continue;

        const std::vector<float> childValues = TraverseBestResponse(
            problem, strategy, node.GetBettingEdge(actionIndex).NextNode(), responderHand, opponentHands, childReach, responder
        );
        for (std::size_t handIndex = 0; handIndex < opponentHands.size(); ++handIndex)
        {
            if (opponentReach[handIndex] > 0.0f)
            {
                const float* probabilities = opponentStrategies[handIndex];
                const float probability = probabilities ? probabilities[actionIndex] : uniformProbability;
                values[handIndex] += probability * childValues[handIndex];
            }
        }
    }
    return values;
}

float ComputeBestResponseEv(const SolveProblem& problem, const StrategySnapshot& strategy, core::PlayerId responder)
{
    const core::Range::Table& myRange = problem.ranges.For(responder).Entries();
    const core::Range::Table& opponentRange = problem.ranges.For(responder.Other()).Entries();
    const game::CompiledGame& game = *problem.game;
    const core::Board& rootBoard = game.GetNode(game.Root()).State().board;

    double totalValue = 0.0;
    double jointRangeWeight = 0.0;
    for (const auto& [myHand, myWeight] : myRange)
    {
        if (myWeight <= 0.0f || core::Overlaps(myHand, rootBoard))
            continue;

        std::vector<core::HoleCards> opponentHands;
        std::vector<float> opponentWeights;
        double totalOpponentWeight = 0.0;
        for (const auto& [opponentHand, opponentWeight] : opponentRange)
        {
            if (opponentWeight <= 0.0f || core::Overlaps(myHand, opponentHand) || core::Overlaps(opponentHand, rootBoard))
                continue;

            opponentHands.push_back(opponentHand);
            opponentWeights.push_back(opponentWeight);
            totalOpponentWeight += opponentWeight;
        }

        if (opponentHands.empty())
            continue;

        const double jointMass = static_cast<double>(myWeight) * totalOpponentWeight;
        jointRangeWeight += jointMass;
        for (float& opponentWeight : opponentWeights)
            opponentWeight = static_cast<float>(opponentWeight / totalOpponentWeight);

        const std::vector<float> values =
            TraverseBestResponse(problem, strategy, game.Root(), myHand, opponentHands, opponentWeights, responder);
        for (std::size_t handIndex = 0; handIndex < opponentHands.size(); ++handIndex)
            totalValue += jointMass * opponentWeights[handIndex] * values[handIndex];
    }

    if (jointRangeWeight <= 0.0)
        throw std::runtime_error("No valid private hand pairs for exploitability evaluation");
    return static_cast<float>(totalValue / jointRangeWeight);
}
} // namespace

ExploitabilityMetrics EvaluateExploitability(const SolveProblem& problem, const StrategySnapshot& strategy)
{
    if (!problem.game || problem.game.get() != &strategy.Game())
        throw std::invalid_argument("Exploitability strategy belongs to a different compiled game");
    const float player0BestResponseEv = ComputeBestResponseEv(problem, strategy, core::PlayerId::Player0());
    const float player1BestResponseEv = ComputeBestResponseEv(problem, strategy, core::PlayerId::Player1());
    return {
        player0BestResponseEv,
        player1BestResponseEv,
        (player0BestResponseEv + player1BestResponseEv) / 2.0f,
    };
}
} // namespace solver::engine
