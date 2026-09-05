#include "engine/CpuEscfrSession.h"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
namespace
{
struct HandPair
{
    core::HoleCards player0Hand;
    core::HoleCards player1Hand;
};
} // namespace

CpuEscfrSession::CpuEscfrSession(std::shared_ptr<const SolveProblem> problem)
    : problem_(std::move(problem)), infoSetsByNode_(problem_ && problem_->game ? problem_->game->NodeCount() : 0)
{
    if (!problem_ || !problem_->game)
        throw std::invalid_argument("CPU ESCFR session requires a solve problem");
}

void CpuEscfrSession::Run(int iterations, const ProgressCallback& progressCallback)
{
    if (iterations <= 0)
        throw std::invalid_argument("Solve iterations must be positive");

    const game::CompiledGame& game = *problem_->game;
    const core::Board& rootBoard = game.GetNode(game.Root()).State().board;
    const core::Range::Table& player0Range = problem_->ranges.For(core::PlayerId::Player0()).Entries();
    const core::Range::Table& player1Range = problem_->ranges.For(core::PlayerId::Player1()).Entries();

    std::vector<HandPair> handPairs;
    std::vector<double> handPairWeights;
    for (const auto& [player0Hand, player0Weight] : player0Range)
    {
        if (player0Weight <= 0.0f || core::Overlaps(player0Hand, rootBoard))
            continue;

        for (const auto& [player1Hand, player1Weight] : player1Range)
        {
            if (player1Weight <= 0.0f || core::Overlaps(player0Hand, player1Hand) || core::Overlaps(player1Hand, rootBoard))
                continue;

            handPairs.push_back({player0Hand, player1Hand});
            handPairWeights.push_back(static_cast<double>(player0Weight) * player1Weight);
        }
    }

    if (handPairs.empty())
        throw std::runtime_error("No valid private hand pairs after applying range weights and blockers");

    std::discrete_distribution<std::size_t> handPairPicker(handPairWeights.begin(), handPairWeights.end());
    auto lastProgressUpdate = std::chrono::steady_clock::now();
    const auto solveStart = std::chrono::steady_clock::now();

    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const core::PlayerId updatingPlayer = completedIterations_ % 2 == 0 ? core::PlayerId::Player0() : core::PlayerId::Player1();
        const HandPair& handPair = handPairs[handPairPicker(rng_)];

        SampleTraverse(game.Root(), updatingPlayer, handPair.player0Hand, handPair.player1Hand);
        ++completedIterations_;

        const int completedInRun = iteration + 1;
        const bool solvingComplete = completedInRun == iterations;
        if (progressCallback && (solvingComplete || completedInRun % 1000 == 0))
        {
            const auto now = std::chrono::steady_clock::now();
            if (solvingComplete || now - lastProgressUpdate >= std::chrono::milliseconds(250))
            {
                progressCallback(completedIterations_);
                lastProgressUpdate = now;
            }
        }
    }

    trainingTimeSeconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - solveStart).count();
}

StrategySnapshot CpuEscfrSession::ExportStrategy() const
{
    const game::CompiledGame& game = *problem_->game;
    std::vector<StrategyEntry> entries;
    for (std::size_t nodeIndex = 0; nodeIndex < game.NodeCount(); ++nodeIndex)
    {
        const game::NodeId nodeId(static_cast<std::int32_t>(nodeIndex));
        const std::size_t actionCount = game.GetNode(nodeId).BettingEdgeCount();
        if (actionCount == 0)
            continue;

        for (const auto& [hand, infoSet] : infoSetsByNode_[nodeIndex])
        {
            std::vector<float> averageStrategy(actionCount, 0.0f);
            const float sum = std::accumulate(infoSet.strategySum.begin(), infoSet.strategySum.end(), 0.0f);
            if (sum > 0.0f)
            {
                for (std::size_t actionIndex = 0; actionIndex < infoSet.strategySum.size(); ++actionIndex)
                    averageStrategy[actionIndex] = infoSet.strategySum[actionIndex] / sum;
            }
            else
            {
                const float uniformProbability = 1.0f / static_cast<float>(actionCount);
                std::fill(averageStrategy.begin(), averageStrategy.end(), uniformProbability);
            }
            entries.push_back({
                {nodeId, hand},
                std::move(averageStrategy),
            });
        }
    }

    return StrategySnapshot(problem_->game, std::move(entries));
}

float CpuEscfrSession::SampleTraverse(
    game::NodeId nodeId,
    core::PlayerId updatingPlayer,
    core::HoleCards player0Hand,
    core::HoleCards player1Hand
)
{
    const game::GameNode& node = problem_->game->GetNode(nodeId);
    if (node.Kind() == game::NodeKind::Terminal)
    {
        const auto [player0Value, player1Value] = problem_->game->CalculateZeroSumUtility(nodeId, player0Hand, player1Hand);
        return updatingPlayer == core::PlayerId::Player0() ? player0Value : player1Value;
    }

    if (node.Kind() == game::NodeKind::Chance)
    {
        if (node.ChanceOutcomeCount() == 0)
            throw std::runtime_error("Chance node has no legal outcomes");

        std::uniform_int_distribution<std::size_t> outcomePicker(0, node.ChanceOutcomeCount() - 1);
        while (true)
        {
            const game::ChanceOutcome& outcome = node.GetChanceOutcome(outcomePicker(rng_));
            if (core::Contains(player0Hand, outcome.DealtCard()) || core::Contains(player1Hand, outcome.DealtCard()))
                continue;

            return SampleTraverse(outcome.NextNode(), updatingPlayer, player0Hand, player1Hand);
        }
    }

    const core::PlayerId actingPlayer = node.State().playerToAct;
    const bool isUpdating = actingPlayer == updatingPlayer;
    const core::HoleCards actingHand = actingPlayer == core::PlayerId::Player0() ? player0Hand : player1Hand;
    InfoSetState& infoSet = GetInfoSet(nodeId, actingHand);

    float regretSum = 0.0f;
    for (const float regret : infoSet.regrets)
        regretSum += std::max(0.0f, regret);

    const float uniformProbability = 1.0f / static_cast<float>(node.BettingEdgeCount());
    const auto actionProbability = [&](std::size_t actionIndex)
    { return regretSum > 0.0f ? std::max(0.0f, infoSet.regrets[actionIndex]) / regretSum : uniformProbability; };

    if (!isUpdating)
    {
        std::uniform_real_distribution<float> probabilityPicker(0.0f, 1.0f);
        const float targetProbability = probabilityPicker(rng_);
        float cumulativeProbability = 0.0f;
        std::size_t selectedActionIndex = node.BettingEdgeCount() - 1;
        bool actionSelected = false;
        for (std::size_t actionIndex = 0; actionIndex < node.BettingEdgeCount(); ++actionIndex)
        {
            const float probability = actionProbability(actionIndex);
            infoSet.strategySum[actionIndex] += probability;
            cumulativeProbability += probability;
            if (!actionSelected && targetProbability < cumulativeProbability)
            {
                selectedActionIndex = actionIndex;
                actionSelected = true;
            }
        }

        return SampleTraverse(node.GetBettingEdge(selectedActionIndex).NextNode(), updatingPlayer, player0Hand, player1Hand);
    }

    float nodeValue = 0.0f;
    for (std::size_t actionIndex = 0; actionIndex < node.BettingEdgeCount(); ++actionIndex)
    {
        const float probability = actionProbability(actionIndex);
        const float value = SampleTraverse(node.GetBettingEdge(actionIndex).NextNode(), updatingPlayer, player0Hand, player1Hand);
        nodeValue += probability * value;
        infoSet.regrets[actionIndex] += value;
    }

    for (std::size_t actionIndex = 0; actionIndex < node.BettingEdgeCount(); ++actionIndex)
        infoSet.regrets[actionIndex] -= nodeValue;

    return nodeValue;
}

CpuEscfrSession::InfoSetState& CpuEscfrSession::GetInfoSet(game::NodeId node, core::HoleCards hand)
{
    InfoSetState& infoSet = infoSetsByNode_[node.Value()][hand];
    if (infoSet.regrets.empty())
    {
        const std::size_t actionCount = problem_->game->GetNode(node).BettingEdgeCount();
        infoSet.regrets.resize(actionCount, 0.0f);
        infoSet.strategySum.resize(actionCount, 0.0f);
    }
    return infoSet;
}
} // namespace solver::engine
