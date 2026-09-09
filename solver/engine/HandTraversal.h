#pragma once

#include "engine/SolveProblem.h"
#include "engine/StrategySnapshot.h"
#include <array>
#include <cstdint>
#include <vector>

namespace solver::engine
{
// Concrete CPU layout and hand-vector kernels shared by training and snapshot evaluation.
// Local node indices belong to this subtree; Node::id retains the complete game history.
struct HandTraversal
{
    // Maximum distinct exact private hands in a 52-card deck (52 choose 2).
    static constexpr std::size_t kMaxHands = 52 * 51 / 2;

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
        game::NodeId id;
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

    std::array<std::vector<Hand>, 2> hands;
    std::vector<Node> nodes;
    std::vector<std::uint32_t> children;
    std::vector<std::uint64_t> dealtCardMasks;
    std::vector<std::vector<std::uint32_t>> levels;
    std::vector<std::uint32_t> terminals;
    std::vector<std::array<std::vector<RankedHand>, 2>> rankRows;

    std::size_t strategySize = 0;
    float rootHalfPot = 0.0f;

    HandTraversal(const SolveProblem& problem, game::NodeId root);
    std::vector<float> LoadStrategy(const StrategySnapshot& strategy) const;
    // Weights proportional to opponent input weights times ancestor action probabilities.
    // The common chance-path factor cancels when conditioning on this subtree's root.
    std::vector<double> OpponentReachAtRoot(const StrategySnapshot& strategy, std::size_t opponentPlayer) const;
    std::vector<double> CompatibleMasses(std::size_t player, const double* opponentReach) const;
    // Training can omit terminal own reach: only opponent reach is read at terminals.
    // Omitted rows remain untouched; every consumer of terminal reach must include them.
    void PropagateReach(
        std::uint32_t node,
        std::size_t player,
        bool includeChance,
        const float* strategy,
        const double* parent,
        double* reaches,
        bool includeTerminals = true
    ) const;
    // Utilities are zero-sum relative to this subtree's root. Divisors stay fixed for the entire pass.
    void EvaluateTerminal(std::uint32_t node, std::size_t player, const double* opponentReach, const double* divisors, float* values) const;
    void BackUp(std::uint32_t node, std::size_t player, const float* strategy, bool bestResponse, float* values) const;
};
} // namespace solver::engine
