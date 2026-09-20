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
    struct Workspace
    {
        std::array<std::vector<float>, 2> reach;
        std::vector<float> childValues;
        std::vector<float> accumulated;
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
        std::uint64_t flopOutcomesBytes;
    };
    static StorageEstimate EstimateStorage(const game::CompiledGame& game, const std::array<std::size_t, 2>& handCounts);

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
    using RankOrder = HandTraversalData::RankOrder;
    using ChanceGroup = HandTraversalData::ChanceGroup;
    using ChanceTask = HandTraversalData::ChanceTask;
    using FlopOutcomes = HandTraversalData::FlopOutcomes;
    struct WorkspaceSize
    {
        std::array<std::size_t, 2> reach;
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
    const std::array<std::vector<std::uint64_t>, 2>& handMasks;
    const std::vector<std::uint32_t>& children;
    const std::vector<std::uint64_t>& dealtCardMasks;
    const std::vector<std::array<RankOrder, 2>>& rankRows;
    const std::array<int, kMaxHands>& rowsByRunout;
    const std::size_t& maxDepth;

    // Only the entry points construct policy combinations. Every read-only path
    // uses exact runout accumulation, including checkpoints on a training layout.
    struct WalkContext
    {
        std::size_t player;
        const float* divisors;
        const StrategySnapshot* strategy = nullptr;
        const float* strategySums = nullptr;
        TrainState* train = nullptr;
        bool bestResponse = false;
        bool useRunoutCache = false;
    };
    std::vector<float> EvaluateHands(const WalkContext& context, const std::vector<float>& opponentReach) const;
    // Required entry policies stay in a row per depth. Training supplies a
    // cursor only when consuming completed chance tasks in preorder.
    void Walk(
        std::uint32_t node,
        const WalkContext& context,
        Workspace& workspace,
        std::size_t depth,
        float* values,
        std::size_t* parallelCursor = nullptr
    ) const;
    const std::vector<ChanceGroup>& chanceGroups_;
    const std::vector<ChanceTask>& chanceTasks_;
    const std::vector<FlopOutcomes>& flopOutcomes_;

    void MatchRegrets(const Node& node, const TrainState& train, float* current) const;
    void EvaluateFlopRunout(const Node& node, std::size_t player, const float* opponentReach, const float* divisors, float* values) const;
    void PropagateChild(
        std::uint32_t node,
        std::size_t action,
        std::size_t player,
        bool includeChance,
        const float* strategy,
        const float* parent,
        float* child
    ) const;
    void EvaluateTerminal(const Node& node, std::size_t player, const float* opponentReach, const float* divisors, float* values) const;
    void EvaluateRunout(
        Node node,
        std::size_t player,
        const float* opponentReach,
        const float* divisors,
        float* values,
        bool useCache
    ) const;
};
} // namespace solver::engine
