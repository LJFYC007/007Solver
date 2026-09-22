#pragma once

#include "engine/SolveProblem.h"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace solver::engine
{
// Immutable hand indices and rank tables for one problem's ranges and exact board.
// Betting history, node utilities and strategy storage do not belong here.
struct HandBoardData
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

    std::shared_ptr<const game::CompiledGame> game;
    core::Board board;
    std::array<std::vector<Hand>, 2> hands;
    std::array<std::vector<std::uint64_t>, 2> handMasks;
    std::vector<std::array<RankOrder, 2>> rankRows;
    std::array<int, kMaxHands> rowsByRunout;

    HandBoardData(const SolveProblem& problem, game::NodeId root);
};
} // namespace solver::engine
