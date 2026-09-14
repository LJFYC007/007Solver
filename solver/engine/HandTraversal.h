#pragma once

#include "engine/SolveProblem.h"
#include "engine/StrategySnapshot.h"
#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace solver::engine
{
// CPU hand-vector kernels. Training state remains owned by CpuDcfrSession.
struct HandTraversal
{
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
        core::Board board;
        int rankRow = -1;
        bool forcedRunout = false;
        std::array<float, 3> utilities{};
    };
    struct RankedHand
    {
        std::uint16_t rank;
        std::uint16_t hand;
    };
    // One depth-first stack per worker, reused across all subtrees and iterations.
    struct Workspace
    {
        std::array<std::vector<double>, 2> reach;
        std::vector<float> childValues;
        std::vector<double> accumulated;
        std::vector<float> parallelValues;
        std::vector<float> strategies;
    };
    using Update = std::function<void(std::uint32_t, const double*, const float*, const float*, const float*)>;
    std::array<std::vector<Hand>, 2> hands;
    std::vector<Node> nodes;
    std::vector<std::uint32_t> children;
    std::vector<std::uint64_t> dealtCardMasks;
    std::vector<std::array<std::vector<RankedHand>, 2>> rankRows;
    std::array<int, 1326> rowsByRunout;
    std::size_t strategySize = 0;
    std::size_t maxDepth = 1;
    std::size_t maxActions = 1;
    float rootHalfPot = 0.0f;

    HandTraversal(const SolveProblem& problem, game::NodeId root);
    Workspace MakeWorkspace(bool parallel = false) const;
    std::vector<double> OpponentReachAtRoot(const StrategySnapshot& strategy, std::size_t opponentPlayer) const;
    std::vector<double> CompatibleMasses(std::size_t player, const double* opponentReach) const;
    // Training supplies regrets and an update callback; evaluation supplies a fixed
    // snapshot. Both paths retain the entry policy in a workspace row at each depth.
    void Walk(
        std::uint32_t node,
        std::size_t player,
        const StrategySnapshot* strategy,
        const double* divisors,
        bool bestResponse,
        Workspace& workspace,
        std::size_t depth,
        float* values,
        const Update& update = {},
        std::vector<Workspace>* workers = nullptr,
        const float* regrets = nullptr
    ) const;

private:
    void PropagateChild(
        std::uint32_t node,
        std::size_t action,
        std::size_t player,
        bool includeChance,
        const float* strategy,
        const double* parent,
        double* child
    ) const;
    void EvaluateTerminal(const Node& node, std::size_t player, const double* opponentReach, const double* divisors, float* values) const;
    void EvaluateRunout(Node node, std::size_t player, const double* opponentReach, const double* divisors, float* values) const;
};
} // namespace solver::engine
