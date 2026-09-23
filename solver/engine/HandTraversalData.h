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
    std::vector<std::uint64_t> dealtCardMasks;
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

// Resident DCFR state in the strategySize layout shared by both backends.
struct TrainingState
{
    std::vector<float> regrets;
    std::vector<float> strategySums;
};
} // namespace solver::engine
