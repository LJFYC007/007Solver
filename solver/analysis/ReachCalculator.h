#pragma once

#include "core/Range.h"
#include "engine/SolveResult.h"
#include <cstddef>
#include <map>
#include <vector>

namespace solver::analysis
{
class ReachCalculator
{
public:
    using HandWeights = std::map<core::HoleCards, float>;

    struct JointReach
    {
        core::HoleCards player0Hand;
        core::HoleCards player1Hand;
        float jointReachMass;
    };

    struct PlayerOwnReachWeights
    {
        HandWeights player0;
        HandWeights player1;
    };

    using JointReachMasses = std::vector<JointReach>;

    struct NodeReach
    {
        JointReachMasses jointReachMasses;
        PlayerOwnReachWeights ownReachWeights;
    };

    explicit ReachCalculator(const engine::SolveResult& result);

    const NodeReach& ReachFor(game::NodeId nodeId);
    HandWeights BuildMarginalReachMasses(const JointReachMasses& jointReachMasses, core::PlayerId player) const;

private:
    const engine::SolveResult& result_;
    std::map<game::NodeId, NodeReach> reachByNode_;

    JointReachMasses BuildInitialJointReachMasses(const core::RangeSet& ranges, const core::Board& board) const;
    JointReachMasses PropagateActionReach(game::NodeId nodeId, std::size_t childIndex, const JointReachMasses& jointReachMasses) const;
    PlayerOwnReachWeights PropagateOwnReach(game::NodeId nodeId, std::size_t childIndex, const PlayerOwnReachWeights& ownReach) const;
    JointReachMasses PropagateChanceReach(const JointReachMasses& jointReachMasses, core::Card dealtCard, int legalOutcomeCount) const;
};
} // namespace solver::analysis
