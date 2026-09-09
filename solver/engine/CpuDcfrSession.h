#pragma once

#include "engine/SolveProblem.h"
#include "engine/StrategySnapshot.h"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace solver::engine
{
class CpuDcfrSession
{
public:
    // Zero workers selects the OpenMP runtime's default team size.
    explicit CpuDcfrSession(std::shared_ptr<const SolveProblem> problem, int workers = 0);

    // One iteration is a full update of one player, alternating across successive runs.
    void Run(int iterations, const std::function<void(int completedIterations)>& progressCallback = {});
    StrategySnapshot ExportStrategy() const;
    int CompletedIterations() const { return completedIterations_; }
    double TrainingTimeSeconds() const { return trainingTimeSeconds_; }
    int WorkerCount() const { return workerCount_; }

private:
    struct Hand
    {
        core::HoleCards cards;
        double weight;
        std::uint64_t mask;
        std::array<std::uint8_t, 2> cardIndices;
        int matchingOpponent = -1;
        double opponentMass = 0.0;
    };

    struct Node
    {
        game::NodeKind kind;
        std::size_t actor;
        std::size_t childOffset;
        std::size_t childCount;
        std::size_t strategyOffset;
        std::uint64_t boardMask;
        int rankRow = -1;
        // Player-0 utility for a win, tie and loss; folds use the first entry.
        std::array<float, 3> utilities{};
    };

    struct RankedHand
    {
        std::uint16_t rank;
        std::uint16_t hand;
    };

    std::shared_ptr<const SolveProblem> problem_;
    std::array<std::vector<Hand>, 2> hands_;
    std::vector<Node> nodes_;
    std::vector<std::uint32_t> children_;
    std::vector<std::uint64_t> dealtCardMasks_;
    std::vector<std::vector<std::uint32_t>> levels_;
    std::vector<std::uint32_t> terminals_;
    std::vector<std::array<std::vector<RankedHand>, 2>> rankRows_;
    // Action-major rows, with a fixed root-hand stride for the acting player.
    std::vector<float> regrets_;
    std::vector<float> strategySums_;
    std::vector<float> strategies_;
    std::array<std::vector<double>, 2> reach_;
    std::vector<float> values_;
    std::size_t infoSetCount_ = 0;
    int workerCount_;
    int completedIterations_ = 0;
    double trainingTimeSeconds_ = 0.0;

    void PropagateReach(std::uint32_t node, std::size_t updatingPlayer);
    void EvaluateTerminal(std::uint32_t node, std::size_t updatingPlayer);
    void BackUp(std::uint32_t node, std::size_t updatingPlayer, float positiveDiscount, float averageDiscount);
};
} // namespace solver::engine
