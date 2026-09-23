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
    // Hands holding each card: cardLists[cardOffsets[c], cardOffsets[c + 1]).
    struct CardLists
    {
        std::array<std::uint16_t, 53> cardOffsets{};
        std::vector<std::uint16_t> cardLists;
    };
    // Rank-major showdown table, by ranked position unless noted; board-blocked hands are absent.
    struct RankOrder
    {
        std::vector<std::uint16_t> ranks;
        std::vector<std::uint16_t> hands;
        // Opponent rank positions delimiting strictly weaker and stronger hands.
        std::vector<std::uint16_t> lowerBounds;
        std::vector<std::uint16_t> upperBounds;
        // This player's hands (hand indices) holding each card, in rank order.
        CardLists holders;
        // How many opponent hands holding this hand's first/second card rank strictly below,
        // then at most, the hand: below0 | through0 << 8 | below1 << 16 | through1 << 24.
        // Fewer than 256 hands hold any card.
        std::vector<std::uint32_t> blockers;
        // Where each held card's run over the opponent's holders starts, offset by the card
        // index for a leading zero, and its length: offset | length << 16.
        std::vector<std::uint32_t> runs0;
        std::vector<std::uint32_t> runs1;
    };

    std::shared_ptr<const game::CompiledGame> game;
    core::Board board;
    std::array<std::vector<Hand>, 2> hands;
    std::array<std::vector<std::uint64_t>, 2> handMasks;
    // Each player's hands holding each card, in hand order, for board-independent folds.
    std::array<CardLists, 2> holders;
    // Per card, one factor per hand: zero when the hand holds the card, otherwise one.
    std::array<std::vector<float>, 2> cardFactors;
    std::vector<std::array<RankOrder, 2>> rankRows;
    std::array<int, kMaxHands> rowsByRunout;

    HandBoardData(const SolveProblem& problem, game::NodeId root);
    int RankRow(const core::Board& river) const { return rowsByRunout[core::CardPairIndex(river.CardAt(3), river.CardAt(4))]; }
};
} // namespace solver::engine
