#include "engine/CpuEscfrSession.h"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
CpuEscfrSession::CpuEscfrSession(std::shared_ptr<const SolveProblem> problem)
    : problem_(std::move(problem)), pageRowsByNode_(problem_ && problem_->game ? problem_->game->NodeCount() : 0, kMissingOffset)
{
    if (!problem_ || !problem_->game)
        throw std::invalid_argument("CPU ESCFR session requires a solve problem");

    const core::Board& board = problem_->game->GetNode(problem_->game->Root()).State().board;
    for (std::size_t player = 0; player < hands_.size(); ++player)
    {
        for (const auto& [hand, weight] : problem_->ranges.For(core::PlayerId(static_cast<std::uint8_t>(player))).Entries())
        {
            if (weight > 0.0f && !core::Overlaps(hand, board))
                hands_[player].push_back({hand, weight});
        }
    }
}

void CpuEscfrSession::Run(int iterations, const ProgressCallback& progressCallback)
{
    if (iterations <= 0)
        throw std::invalid_argument("Solve iterations must be positive");

    const game::CompiledGame& game = *problem_->game;
    std::vector<HandPair> handPairs;
    std::vector<double> handPairWeights;
    for (std::size_t player0Index = 0; player0Index < hands_[0].size(); ++player0Index)
    {
        const RangeHand& player0 = hands_[0][player0Index];
        for (std::size_t player1Index = 0; player1Index < hands_[1].size(); ++player1Index)
        {
            const RangeHand& player1 = hands_[1][player1Index];
            if (core::Overlaps(player0.cards, player1.cards))
                continue;

            handPairs.push_back({player0Index, player1Index});
            handPairWeights.push_back(static_cast<double>(player0.weight) * player1.weight);
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

        SampleTraverse(game.Root(), updatingPlayer, handPair);
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
    std::vector<game::InfoSetKey> infoSets;
    std::vector<float> probabilities;
    infoSets.reserve(visitedInfoSetCount_);
    probabilities.reserve(strategySums_.size());
    for (std::size_t nodeIndex = 0; nodeIndex < game.NodeCount(); ++nodeIndex)
    {
        const std::uint32_t rowOffset = pageRowsByNode_[nodeIndex];
        if (rowOffset == kMissingOffset)
            continue;

        const game::NodeId nodeId(static_cast<std::int32_t>(nodeIndex));
        const game::GameNode& node = game.GetNode(nodeId);
        const std::size_t actionCount = node.BettingEdgeCount();
        const std::vector<RangeHand>& hands = hands_[node.State().playerToAct.Index()];

        for (std::size_t handIndex = 0; handIndex < hands.size(); ++handIndex)
        {
            const std::uint32_t pageOffset = handPageOffsets_[static_cast<std::size_t>(rowOffset) + handIndex / kHandsPerPage];
            if (pageOffset == kMissingOffset)
                continue;

            const std::uint32_t valueOffset = infoSetOffsets_[static_cast<std::size_t>(pageOffset) + handIndex % kHandsPerPage];
            if (valueOffset == kMissingOffset)
                continue;

            infoSets.push_back({nodeId, hands[handIndex].cards});
            const float* strategySum = strategySums_.data() + valueOffset;
            const float sum = std::accumulate(strategySum, strategySum + actionCount, 0.0f);
            if (sum > 0.0f)
            {
                for (std::size_t actionIndex = 0; actionIndex < actionCount; ++actionIndex)
                    probabilities.push_back(strategySum[actionIndex] / sum);
            }
            else
            {
                const float uniformProbability = 1.0f / static_cast<float>(actionCount);
                probabilities.insert(probabilities.end(), actionCount, uniformProbability);
            }
        }
    }

    return StrategySnapshot(problem_->game, infoSets, std::move(probabilities));
}

float CpuEscfrSession::SampleTraverse(game::NodeId nodeId, core::PlayerId updatingPlayer, const HandPair& handPair)
{
    const game::GameNode& node = problem_->game->GetNode(nodeId);
    const core::HoleCards player0Hand = hands_[0][handPair.player0Index].cards;
    const core::HoleCards player1Hand = hands_[1][handPair.player1Index].cards;
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

            return SampleTraverse(outcome.NextNode(), updatingPlayer, handPair);
        }
    }

    const core::PlayerId actingPlayer = node.State().playerToAct;
    const bool isUpdating = actingPlayer == updatingPlayer;
    const std::size_t handIndex = actingPlayer == core::PlayerId::Player0() ? handPair.player0Index : handPair.player1Index;
    // Recursive visits can grow the buffers; retain an offset rather than a pointer into them.
    const std::uint32_t valueOffset = GetInfoSetOffset(nodeId, handIndex);

    float regretSum = 0.0f;
    for (std::size_t actionIndex = 0; actionIndex < node.BettingEdgeCount(); ++actionIndex)
        regretSum += std::max(0.0f, regrets_[valueOffset + actionIndex]);

    const float uniformProbability = 1.0f / static_cast<float>(node.BettingEdgeCount());
    const auto actionProbability = [&](std::size_t actionIndex)
    { return regretSum > 0.0f ? std::max(0.0f, regrets_[valueOffset + actionIndex]) / regretSum : uniformProbability; };

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
            strategySums_[valueOffset + actionIndex] += probability;
            cumulativeProbability += probability;
            if (!actionSelected && targetProbability < cumulativeProbability)
            {
                selectedActionIndex = actionIndex;
                actionSelected = true;
            }
        }

        return SampleTraverse(node.GetBettingEdge(selectedActionIndex).NextNode(), updatingPlayer, handPair);
    }

    float nodeValue = 0.0f;
    for (std::size_t actionIndex = 0; actionIndex < node.BettingEdgeCount(); ++actionIndex)
    {
        const float probability = actionProbability(actionIndex);
        const float value = SampleTraverse(node.GetBettingEdge(actionIndex).NextNode(), updatingPlayer, handPair);
        nodeValue += probability * value;
        regrets_[valueOffset + actionIndex] += value;
    }

    for (std::size_t actionIndex = 0; actionIndex < node.BettingEdgeCount(); ++actionIndex)
        regrets_[valueOffset + actionIndex] -= nodeValue;

    return nodeValue;
}

std::uint32_t CpuEscfrSession::GetInfoSetOffset(game::NodeId nodeId, std::size_t handIndex)
{
    const game::GameNode& node = problem_->game->GetNode(nodeId);
    const std::size_t handCount = hands_[node.State().playerToAct.Index()].size();
    std::uint32_t& rowOffset = pageRowsByNode_[nodeId.Value()];
    if (rowOffset == kMissingOffset)
    {
        const std::size_t pageCount = (handCount + kHandsPerPage - 1) / kHandsPerPage;
        const std::size_t nextRow = handPageOffsets_.size();
        if (pageCount > static_cast<std::size_t>(kMissingOffset) - nextRow)
            throw std::length_error("CPU hand page index exceeds 32-bit offset capacity");
        handPageOffsets_.resize(nextRow + pageCount, kMissingOffset);
        rowOffset = static_cast<std::uint32_t>(nextRow);
    }

    std::uint32_t& pageOffset = handPageOffsets_[static_cast<std::size_t>(rowOffset) + handIndex / kHandsPerPage];
    if (pageOffset == kMissingOffset)
    {
        const std::size_t pageSize = std::min(kHandsPerPage, handCount - handIndex / kHandsPerPage * kHandsPerPage);
        const std::size_t nextPage = infoSetOffsets_.size();
        if (pageSize > static_cast<std::size_t>(kMissingOffset) - nextPage)
            throw std::length_error("CPU infoset index exceeds 32-bit offset capacity");
        infoSetOffsets_.resize(nextPage + pageSize, kMissingOffset);
        pageOffset = static_cast<std::uint32_t>(nextPage);
    }

    const std::size_t slot = static_cast<std::size_t>(pageOffset) + handIndex % kHandsPerPage;
    if (infoSetOffsets_[slot] == kMissingOffset)
    {
        const std::size_t actionCount = node.BettingEdgeCount();
        const std::size_t valueOffset = regrets_.size();
        if (actionCount > static_cast<std::size_t>(kMissingOffset) - valueOffset)
            throw std::length_error("CPU training data exceeds 32-bit offset capacity");
        regrets_.resize(valueOffset + actionCount, 0.0f);
        strategySums_.resize(valueOffset + actionCount, 0.0f);
        infoSetOffsets_[slot] = static_cast<std::uint32_t>(valueOffset);
        ++visitedInfoSetCount_;
    }
    return infoSetOffsets_[slot];
}
} // namespace solver::engine
