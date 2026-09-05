#pragma once

#include "core/Card.h"
#include "core/PokerTypes.h"
#include "engine/SolveProblem.h"
#include "engine/StrategySnapshot.h"
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace solver::engine
{
using ProgressCallback = std::function<void(int completedIterations)>;

class CpuEscfrSession
{
public:
    explicit CpuEscfrSession(std::shared_ptr<const SolveProblem> problem);

    // Each iteration updates one player; successive runs continue the same training state.
    void Run(int iterations, const ProgressCallback& progressCallback = {});
    StrategySnapshot ExportStrategy() const;
    int CompletedIterations() const { return completedIterations_; }
    double TrainingTimeSeconds() const { return trainingTimeSeconds_; }

private:
    struct RangeHand
    {
        core::HoleCards cards;
        float weight;
    };

    struct HandPair
    {
        std::size_t player0Index;
        std::size_t player1Index;
    };

    static constexpr std::uint32_t kMissingOffset = std::numeric_limits<std::uint32_t>::max();
    static constexpr std::size_t kHandsPerPage = 32;

    std::shared_ptr<const SolveProblem> problem_;
    std::default_random_engine rng_{42};
    std::array<std::vector<RangeHand>, 2> hands_;
    std::vector<std::uint32_t> pageRowsByNode_;
    std::vector<std::uint32_t> handPageOffsets_;
    std::vector<std::uint32_t> infoSetOffsets_;
    std::vector<float> regrets_;
    std::vector<float> strategySums_;
    std::size_t visitedInfoSetCount_ = 0;
    int completedIterations_ = 0;
    double trainingTimeSeconds_ = 0.0;

    float SampleTraverse(game::NodeId node, core::PlayerId updatingPlayer, const HandPair& handPair);
    std::uint32_t GetInfoSetOffset(game::NodeId node, std::size_t handIndex);
};
} // namespace solver::engine
