#pragma once

#include "engine/HandTraversalData.h"
#include "engine/StrategySnapshot.h"
#include <array>
#include <cstdint>
#include <vector>

namespace solver::engine
{
// CPU hand-vector kernels. Training state remains owned by CpuDcfrSession.
struct HandTraversal
{
private:
    std::shared_ptr<const HandTraversalData> data_;

public:
    using Hand = HandTraversalData::Hand;
    using Node = HandTraversalData::Node;
    static constexpr std::size_t kMaxHands = HandTraversalData::kMaxHands;
    // One depth-first stack per worker, reused across all subtrees and iterations.
    // Every walk propagates only the opponent's reach: one row per depth.
    struct Workspace
    {
        std::vector<float> reach;
        std::vector<float> childValues;
        std::vector<float> accumulated;
        std::vector<float> parallelValues;
        std::vector<std::uint8_t> parallelLive; // whether each chance task's subtree carried opponent reach
        std::vector<float> strategies;
    };
    // Training Walk inlines regret matching and updates from these buffers. The
    // opponent's strategy sums accumulate averageWeight * reach * policy at its
    // decisions; consumers normalize per hand. Stamps hold each node's last unpruned update.
    struct TrainState
    {
        float* regrets = nullptr;
        float* strategySums = nullptr;
        std::uint32_t* stamps = nullptr;
        UpdateWeights weights{};
    };
    enum class Evaluation
    {
        StrategyValue,
        BestResponse,
    };
    // Allocation sizes before constructing the traversal; excludes training/snapshot state.
    struct StorageEstimate
    {
        std::uint64_t fixedBytes;
        std::uint64_t workspaceBytes;
        std::uint64_t parallelValuesBytes;
        std::uint64_t runoutOutcomesBytes;
    };
    static StorageEstimate EstimateStorage(
        const game::CompiledGame& game,
        const std::array<std::size_t, 2>& handCounts,
        bool prepareTraining = false
    );

    const std::array<std::vector<Hand>, 2>& hands;
    const std::vector<Node>& nodes;
    const std::size_t& strategySize;
    const std::size_t& maxActions;
    const float& rootHalfPot;
    // Training amortizes runout caching and the chance-batch plan; analysis leaves them off.
    HandTraversal(const SolveProblem& problem, game::NodeId root, bool prepareTraining = false);
    explicit HandTraversal(std::shared_ptr<const HandTraversalData> data);
    const HandTraversalData& Data() const { return *data_; }
    Workspace MakeWorkspace(bool parallel = false) const;
    void WalkTraining(
        std::size_t player,
        const float* divisors,
        Workspace& workspace,
        float* values,
        std::vector<Workspace>& workers,
        TrainState& train
    ) const;
    std::vector<float> OpponentReachAtRoot(const StrategySnapshot& strategy, std::size_t opponentPlayer) const;
    std::vector<float> CompatibleMasses(std::size_t player, const float* opponentReach) const;
    std::vector<float> EvaluateSnapshot(
        const StrategySnapshot& strategy,
        std::size_t player,
        const std::vector<float>& opponentReach,
        const std::vector<float>& divisors,
        Evaluation evaluation
    ) const;
    std::vector<float> EvaluateAverageBestResponse(
        const float* strategySums,
        std::size_t player,
        const std::vector<float>& opponentReach,
        const std::vector<float>& divisors
    ) const;

private:
    using Kind = HandTraversalData::Kind;
    using RankOrder = HandTraversalData::RankOrder;
    using ChanceGroup = HandTraversalData::ChanceGroup;
    using ChanceTask = HandTraversalData::ChanceTask;
    struct WorkspaceSize
    {
        std::size_t reach;
        std::size_t childValues;
        std::size_t accumulated;
        std::size_t parallelValues;
        std::size_t strategies;

        std::uint64_t Bytes() const;
    };
    static WorkspaceSize SizeWorkspace(
        std::size_t depth,
        std::size_t actions,
        const std::array<std::size_t, 2>& handCounts,
        std::size_t chanceTasks
    );
    const std::vector<std::uint32_t>& children;
    const std::vector<std::uint8_t>& dealtCards;
    const std::vector<std::array<RankOrder, 2>>& rankRows;
    const std::size_t& maxDepth;

    // Only the entry points construct policy combinations. Only training uses the runout
    // outcome rows; read-only paths, including checkpoints, use exact runout accumulation.
    struct WalkContext
    {
        std::size_t player;
        const float* divisors;
        const StrategySnapshot* strategy = nullptr;
        const float* strategySums = nullptr;
        TrainState* train = nullptr;
        bool bestResponse = false;
    };
    std::vector<float> EvaluateHands(const WalkContext& context, const std::vector<float>& opponentReach) const;
    // Required entry policies stay in a row per depth. The opponent's reach points at
    // the row written by the nearest ancestor that changed it, which rewrites that row
    // only after the subtree reading it finishes. Returns whether the subtree carried
    // opponent reach, as the GPU's terminal flags do; subtrees without it are skipped
    // and their acting decisions keep their stamps. Training supplies a cursor only when
    // consuming completed chance tasks in preorder.
    bool Walk(
        std::uint32_t node,
        const WalkContext& context,
        Workspace& workspace,
        std::size_t depth,
        const float* opponentReach,
        float* values,
        std::size_t* parallelCursor = nullptr
    ) const;
    const std::vector<ChanceGroup>& chanceGroups_;
    const std::vector<ChanceTask>& chanceTasks_;
    const std::vector<std::uint32_t>& runoutOutcomes_;

    // Regret matching over the actor's hands. Given the actor's reach at the node, hands
    // without reach get a zero policy and their regrets are not read.
    void MatchRegrets(const Node& node, const TrainState& train, float* current, const float* actorReach) const;
    void EvaluateRunoutOutcomes(
        const Node& node,
        std::size_t player,
        const float* opponentReach,
        const float* divisors,
        float* values
    ) const;
    // Returns parent at another player's decision, whose reach the child shares; otherwise
    // writes and returns child, including the chance probability at chance nodes.
    const float* PropagateChild(
        std::uint32_t node,
        std::size_t action,
        std::size_t player,
        const float* strategy,
        const float* parent,
        float* child
    ) const;
    void EvaluateTerminal(const Node& node, std::size_t player, const float* opponentReach, const float* divisors, float* values) const;
    // Utilities are win/tie/loss for player; recurses over undealt cards to river showdowns.
    void EvaluateRunout(
        const std::array<float, 3>& utilities,
        const core::Board& board,
        std::uint64_t boardMask,
        std::size_t player,
        const float* opponentReach,
        const float* divisors,
        float* values
    ) const;
};
} // namespace solver::engine
