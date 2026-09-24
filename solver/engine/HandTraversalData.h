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
    // Runout rows of player-0 win | loss << 16 counts per (player-0 hand, player-1 hand)
    // pair, exact out of the legal runouts and independent of reach/payoffs: row 0 for
    // flop all-ins (990 runouts) and row card + 1 for turn all-ins on that card (44), as
    // the GPU's outcome rows. The player-0-major rows are followed by player-1-major
    // copies so either player's accumulation sweeps its hands contiguously.
    std::vector<std::uint32_t> runoutOutcomes_;
    static std::size_t RunoutRow(const Node& node)
    {
        return node.board.CardCount() == 3 ? 0 : static_cast<std::size_t>(node.board.CardAt(3).Index()) + 1;
    }
    std::size_t runoutRows = 0; // one past the largest RunoutRow of a forced runout, zero without any

    std::size_t infoSetCount = 0;

    HandTraversalData(const SolveProblem& problem, game::NodeId root, bool prepareTraining = false);
    HandTraversalData(std::shared_ptr<const HandBoardData> tables, game::NodeId root, bool prepareTraining = false);
    StrategySnapshot ExportStrategy(std::vector<float> sums) const;

private:
    void PrepareChanceTasks();
    void PrepareRunoutOutcomes();
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
