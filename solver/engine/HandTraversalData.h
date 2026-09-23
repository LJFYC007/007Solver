#pragma once

#include "engine/HandBoardData.h"
#include "engine/StrategySnapshot.h"
#include "engine/gpu/GpuTypes.h"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace solver::engine
{
// Immutable, range-specific tables shared by recursive CPU and batched GPU execution.
struct HandTraversalData
{
    static constexpr std::size_t kMaxHands = HandBoardData::kMaxHands;
    using Hand = HandBoardData::Hand;
    using RankOrder = HandBoardData::RankOrder;
    // The GPU layout uses the same kinds. Forced runouts are leaves; only showdowns have a rank row.
    using Kind = gpu::NodeKind;
    struct Node
    {
        game::NodeId id;
        Kind kind;
        std::size_t actor;
        std::size_t childOffset;
        std::size_t childCount;
        std::size_t strategyOffset;
        std::uint64_t boardMask;
        core::Board board;
        int rankRow = -1;
        std::array<float, 3> utilities{};

        bool IsLeaf() const { return kind != Kind::Decision && kind != Kind::Chance; }
    };
    std::shared_ptr<const HandBoardData> tables;
    std::vector<Node> nodes;
    std::size_t strategySize = 0;
    std::size_t maxActions = 1;
    float rootHalfPot = 0.0f;

    std::vector<std::uint32_t> children;
    std::vector<std::uint8_t> dealtCards; // card index of each chance edge
    std::size_t maxDepth = 1;

    struct ChanceGroup
    {
        std::uint32_t node;
        std::vector<std::uint32_t> path;
    };
    struct ChanceTask
    {
        std::uint32_t group;
        std::uint32_t action;
    };
    std::vector<ChanceGroup> chanceGroups_;
    std::vector<ChanceTask> chanceTasks_;
    struct FlopOutcomes
    {
        std::uint16_t wins = 0;
        std::uint16_t losses = 0;
    };
    // One player-0-major table for this traversal's flop and exact hand layout.
    // Counts are exact out of C(45, 2) legal runouts, independent of reach/payoffs.
    std::vector<FlopOutcomes> flopOutcomes_;

    std::size_t infoSetCount = 0;

    HandTraversalData(const SolveProblem& problem, game::NodeId root, bool prepareTraining = false);
    HandTraversalData(std::shared_ptr<const HandBoardData> tables, game::NodeId root, bool prepareTraining = false);
    StrategySnapshot ExportStrategy(std::vector<float> sums) const;

private:
    void PrepareChanceTasks();
    void PrepareFlopRunout();
};

// Resident DCFR state in the layouts shared by both backends: regrets and cumulative
// strategies in the strategySize layout, and per traversal node the update index of the
// actor's last unpruned update (zero before the first), which discounts regrets lazily.
struct TrainingState
{
    std::vector<float> regrets;
    std::vector<float> strategySums;
    std::vector<std::uint32_t> stamps;
};

// One player update's DCFR weights. Positive regrets are stored divided by positiveScale,
// the product of t^1.5 / (t^1.5 + 1) over the player's earlier updates, so pruned
// updates need no discount pass; negative regrets are halved once per skipped update.
struct UpdateWeights
{
    std::uint32_t update;  // 1-based index of this update for the updating player
    float positiveScale;   // product of earlier positive discounts
    float positiveInverse; // 1 / positiveScale
    float averageWeight;   // t^2, the weight of this update's reach * policy in the sums
};
} // namespace solver::engine
