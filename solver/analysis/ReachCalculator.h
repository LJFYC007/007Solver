#pragma once

#include "core/Range.h"
#include "engine/SolveResult.h"
#include <cstddef>
#include <map>
#include <optional>
#include <vector>

namespace solver::analysis
{
class ReachCalculator
{
public:
    using HandWeights = std::map<core::HoleCards, float>;

    struct PlayerOwnReachWeights
    {
        HandWeights player0;
        HandWeights player1;
    };

    struct NodeReach
    {
        PlayerOwnReachWeights ownReachWeights;
        float chanceProbability = 1.0f;
    };

    explicit ReachCalculator(const engine::SolveResult& result);

    // Borrows the current path's reach; a later query may invalidate the reference.
    const NodeReach& ReachFor(game::NodeId nodeId);
    HandWeights BuildMarginalReachMasses(const NodeReach& reach, const core::Board& board, core::PlayerId player) const;
    // No value means no supported pair; zero means no card blocks every pair.
    std::optional<std::uint64_t> CommonBlockers(const NodeReach& reach, const core::Board& board) const;

private:
    const engine::SolveResult& result_;
    float rootMass_ = 0.0f;
    struct CachedReach
    {
        game::NodeId node;
        NodeReach reach;
    };
    std::vector<CachedReach> path_;

    PlayerOwnReachWeights PropagateOwnReach(game::NodeId nodeId, std::size_t childIndex, const PlayerOwnReachWeights& ownReach) const;
};
} // namespace solver::analysis
