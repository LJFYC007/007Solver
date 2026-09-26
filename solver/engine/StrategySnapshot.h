#pragma once

#include "core/Card.h"
#include "game/CompiledGame.h"
#include "game/Identifiers.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace solver::engine
{
struct StrategyEntry
{
    game::InfoSetKey infoSet;
    std::vector<float> probabilities;
};

class StrategySnapshot
{
public:
    // Entries must name decision nodes and unique board-compatible hands, with finite,
    // non-negative probabilities per action that sum to one.
    StrategySnapshot(std::shared_ptr<const game::CompiledGame> game, std::vector<StrategyEntry> entries);
    // Exported hand lists (one per actor and board) are negligible beside the probabilities.
    static std::uint64_t EstimateStorageBytes(std::size_t nodes, std::size_t probabilities);

    // Borrows one node's sorted hands and hand-major probabilities from this snapshot.
    // Destruction, move or assignment of the snapshot invalidates the view.
    struct NodeStrategyView
    {
        const core::HoleCards* hands;
        const float* probabilities;
        std::size_t handCount;
        std::size_t actionCount;
    };
    std::optional<NodeStrategyView> FindNodeStrategy(game::NodeId node) const;

    const game::CompiledGame& Game() const { return *game_; }
    // Borrowed probabilities: length is the node's action count; invalidated by destruction, move or assignment.
    const float* FindStrategy(const game::InfoSetKey& infoSet) const;
    // The action must belong to this decision node; missing hands use uniform play.
    float ActionProbability(const game::InfoSetKey& infoSet, std::size_t action) const;
    std::vector<float> StrategyOrUniform(const game::InfoSetKey& infoSet) const;

private:
    friend struct HandTraversalData;

    struct NodeBlock
    {
        game::NodeId node;
        std::size_t handOffset;
        std::size_t probabilityOffset;
        std::size_t handCount;
        std::size_t actionCount;
    };

    // Exports build valid blocks from the traversal layout.
    StrategySnapshot(
        std::shared_ptr<const game::CompiledGame> game,
        std::vector<NodeBlock> nodes,
        std::vector<core::HoleCards> hands,
        std::unique_ptr<float[]> probabilities
    );

    std::shared_ptr<const game::CompiledGame> game_;
    // Sorted by node; each block owns its hand-major probabilities, while blocks may share a
    // sorted hand list.
    std::vector<NodeBlock> nodes_;
    std::vector<core::HoleCards> hands_;
    std::unique_ptr<float[]> probabilities_;
};
} // namespace solver::engine
