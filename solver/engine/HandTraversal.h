#pragma once

#include "engine/SolveProblem.h"
#include "engine/StrategySnapshot.h"
#include <array>
#include <cstdint>
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
    // Rank-major showdown table: sequential sweeps never chase Hand rows.
    struct RankOrder
    {
        std::vector<std::uint16_t> ranks;
        std::vector<std::uint16_t> hands;
        std::vector<std::uint8_t> card0;
        std::vector<std::uint8_t> card1;
        std::vector<std::uint64_t> masks;
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
    // Training Walk inlines regret matching and updates from these buffers.
    struct TrainState
    {
        float* regrets = nullptr;
        float* strategySums = nullptr;
        float positiveDiscount = 0.0f;
        float averageDiscount = 0.0f;
    };
    std::array<std::vector<Hand>, 2> hands;
    std::array<std::vector<std::uint64_t>, 2> handMasks;
    std::vector<Node> nodes;
    std::vector<std::uint32_t> children;
    std::vector<std::uint64_t> dealtCardMasks;
    std::vector<std::array<RankOrder, 2>> rankRows;
    std::array<int, 1326> rowsByRunout;
    std::size_t strategySize = 0;
    std::size_t maxDepth = 1;
    std::size_t maxActions = 1;
    float rootHalfPot = 0.0f;

    // Training amortizes runout caching and the chance-batch plan; analysis leaves them off.
    HandTraversal(const SolveProblem& problem, game::NodeId root, bool prepareTraining = false);
    Workspace MakeWorkspace(bool parallel = false) const;
    void WalkTraining(
        std::size_t player,
        const double* divisors,
        Workspace& workspace,
        float* values,
        std::vector<Workspace>& workers,
        TrainState& train
    ) const;
    std::vector<double> OpponentReachAtRoot(const StrategySnapshot& strategy, std::size_t opponentPlayer) const;
    std::vector<double> CompatibleMasses(std::size_t player, const double* opponentReach) const;
    // Training supplies TrainState; evaluation supplies a snapshot or cumulative
    // strategy sums. All paths retain the entry policy in a row per depth. WalkTraining
    // supplies a cursor to consume its completed chance tasks in preorder.
    void Walk(
        std::uint32_t node,
        std::size_t player,
        const StrategySnapshot* strategy,
        const double* divisors,
        bool bestResponse,
        Workspace& workspace,
        std::size_t depth,
        float* values,
        std::size_t* parallelCursor = nullptr,
        TrainState* train = nullptr,
        const float* strategySums = nullptr
    ) const;

private:
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

    void PrepareChanceTasks();
    void MatchRegrets(const Node& node, const TrainState& train, float* current) const;
    void PrepareFlopRunout();
    void EvaluateFlopRunout(const Node& node, std::size_t player, const double* opponentReach, const double* divisors, float* values) const;
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
    void EvaluateRunout(
        Node node,
        std::size_t player,
        const double* opponentReach,
        const double* divisors,
        float* values,
        bool useCache = true
    ) const;
};
} // namespace solver::engine
