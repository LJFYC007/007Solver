#pragma once

#include "engine/SolveProblem.h"
#include "engine/StrategySnapshot.h"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace solver::engine
{
// Immutable, range-specific tables shared by recursive CPU and batched GPU execution.
struct HandTraversalData
{
    static constexpr std::size_t kMaxHands = 52 * 51 / 2;
    struct Hand
    {
        core::HoleCards cards;
        float weight;
        std::uint64_t mask;
        std::array<std::uint8_t, 2> cardIndices;
        int matchingOpponent = -1;
        float opponentMass = 0.0f;
    };
    struct Node
    {
        game::NodeId id;
        game::NodeKind kind;
        std::size_t actor;
        std::size_t childOffset;
        std::size_t childCount;
        std::size_t strategyOffset;
        std::uint64_t boardMask;
        core::Board board;
        int rankRow = -1;
        bool forcedRunout = false;
        std::array<float, 3> utilities{};
    };
    // Rank-major showdown table: sequential sweeps never chase Hand rows.
    struct RankOrder
    {
        std::vector<std::uint16_t> ranks;
        std::vector<std::uint16_t> hands;
        // Opponent rank positions delimiting strictly weaker and stronger hands.
        std::vector<std::uint16_t> lowerBounds;
        std::vector<std::uint16_t> upperBounds;
        std::vector<std::uint8_t> card0;
        std::vector<std::uint8_t> card1;
        std::vector<std::uint64_t> masks;
    };
    std::array<std::vector<Hand>, 2> hands;
    std::vector<Node> nodes;
    std::size_t strategySize = 0;
    std::size_t maxActions = 1;
    float rootHalfPot = 0.0f;

    std::array<std::vector<std::uint64_t>, 2> handMasks;
    std::vector<std::uint32_t> children;
    std::vector<std::uint64_t> dealtCardMasks;
    std::vector<std::array<RankOrder, 2>> rankRows;
    std::array<int, kMaxHands> rowsByRunout;
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

    std::shared_ptr<const game::CompiledGame> game;
    std::size_t infoSetCount = 0;

    HandTraversalData(const SolveProblem& problem, game::NodeId root, bool prepareTraining = false);
    StrategySnapshot ExportStrategy(std::vector<float> sums) const;

private:
    void PrepareChanceTasks();
    void PrepareFlopRunout();
};
} // namespace solver::engine
