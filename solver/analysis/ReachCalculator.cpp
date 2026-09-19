#include "analysis/ReachCalculator.h"
#include "game/CompiledGame.h"
#include <stdexcept>
#include <utility>

namespace solver::analysis
{
namespace
{
// Heads-up hold'em: each player holds two private cards that chance cannot deal.
constexpr int kHeadsUpHoleCardCount = 4;
} // namespace

ReachCalculator::ReachCalculator(const engine::SolveResult& result) : result_(result)
{
    const engine::SolveProblem& problem = result_.Problem();
    reachByNode_.emplace(
        problem.game->Root(),
        NodeReach{
            BuildInitialJointReachMasses(problem.ranges, problem.game->Spec().initialBoard),
            {problem.ranges.For(core::PlayerId::Player0()).Entries(), problem.ranges.For(core::PlayerId::Player1()).Entries()},
        }
    );
}

const ReachCalculator::NodeReach& ReachCalculator::ReachFor(game::NodeId nodeId)
{
    const auto cached = reachByNode_.find(nodeId);
    if (cached != reachByNode_.end())
        return cached->second;

    const game::CompiledGame& game = *result_.Problem().game;
    const auto parentEdge = *game.GetNode(nodeId).Parent();
    const NodeReach& parentReach = ReachFor(parentEdge.node);
    const game::GameNode& parent = game.GetNode(parentEdge.node);
    NodeReach reach;
    if (parent.Kind() == game::NodeKind::Chance)
    {
        const core::Card dealtCard = parent.GetChanceOutcome(parentEdge.edgeIndex).DealtCard();
        const int legalOutcomeCount = 52 - parent.State().board.CardCount() - kHeadsUpHoleCardCount;
        reach.jointReachMasses = PropagateChanceReach(parentReach.jointReachMasses, dealtCard, legalOutcomeCount);
        reach.ownReachWeights = parentReach.ownReachWeights;
    }
    else
    {
        reach.jointReachMasses = PropagateActionReach(parentEdge.node, parentEdge.edgeIndex, parentReach.jointReachMasses);
        reach.ownReachWeights = PropagateOwnReach(parentEdge.node, parentEdge.edgeIndex, parentReach.ownReachWeights);
    }
    return reachByNode_.emplace(nodeId, std::move(reach)).first->second;
}

ReachCalculator::HandWeights ReachCalculator::BuildMarginalReachMasses(
    const JointReachMasses& jointReachMasses,
    core::PlayerId player
) const
{
    HandWeights marginalReachMasses;
    for (const JointReach& jointReach : jointReachMasses)
    {
        marginalReachMasses[player == core::PlayerId::Player0() ? jointReach.player0Hand : jointReach.player1Hand] +=
            jointReach.jointReachMass;
    }
    return marginalReachMasses;
}

ReachCalculator::JointReachMasses ReachCalculator::BuildInitialJointReachMasses(
    const core::RangeSet& ranges,
    const core::Board& board
) const
{
    JointReachMasses jointReachMasses;
    float totalWeight = 0.0f;
    for (const auto& [player0Hand, player0Weight] : ranges.For(core::PlayerId::Player0()).Entries())
    {
        if (player0Weight <= 0.0f || core::Overlaps(player0Hand, board))
            continue;

        for (const auto& [player1Hand, player1Weight] : ranges.For(core::PlayerId::Player1()).Entries())
        {
            if (player1Weight <= 0.0f || core::Overlaps(player0Hand, player1Hand) || core::Overlaps(player1Hand, board))
                continue;

            jointReachMasses.push_back({player0Hand, player1Hand, 0.0f});
            totalWeight += player0Weight * player1Weight;
        }
    }
    if (totalWeight <= 0.0f)
        throw std::runtime_error("No valid private hand pairs for reach calculation");

    for (JointReach& jointReach : jointReachMasses)
    {
        const float jointWeight = ranges.For(core::PlayerId::Player0()).GetWeight(jointReach.player0Hand) *
                                  ranges.For(core::PlayerId::Player1()).GetWeight(jointReach.player1Hand);
        jointReach.jointReachMass = jointWeight / totalWeight;
    }
    return jointReachMasses;
}

ReachCalculator::JointReachMasses ReachCalculator::PropagateActionReach(
    game::NodeId nodeId,
    std::size_t childIndex,
    const JointReachMasses& jointReachMasses
) const
{
    JointReachMasses propagatedMasses;
    propagatedMasses.reserve(jointReachMasses.size());
    HandWeights actionProbabilities;
    const game::GameNode& node = result_.Problem().game->GetNode(nodeId);
    const core::PlayerId player = node.State().playerToAct;

    for (const JointReach& jointReach : jointReachMasses)
    {
        const core::HoleCards hand = player == core::PlayerId::Player0() ? jointReach.player0Hand : jointReach.player1Hand;
        auto probabilityIt = actionProbabilities.find(hand);
        if (probabilityIt == actionProbabilities.end())
        {
            const float probability = result_.Strategy().ActionProbability({nodeId, hand}, childIndex);
            probabilityIt = actionProbabilities.emplace(hand, probability).first;
        }

        const float jointReachMass = jointReach.jointReachMass * probabilityIt->second;
        if (jointReachMass > 0.0f)
            propagatedMasses.push_back({jointReach.player0Hand, jointReach.player1Hand, jointReachMass});
    }
    return propagatedMasses;
}

ReachCalculator::PlayerOwnReachWeights ReachCalculator::PropagateOwnReach(
    game::NodeId nodeId,
    std::size_t childIndex,
    const PlayerOwnReachWeights& ownReach
) const
{
    PlayerOwnReachWeights propagatedReach = ownReach;
    const game::GameNode& node = result_.Problem().game->GetNode(nodeId);
    HandWeights& actingReach = node.State().playerToAct == core::PlayerId::Player0() ? propagatedReach.player0 : propagatedReach.player1;
    for (auto& [hand, weight] : actingReach)
    {
        if (core::Overlaps(hand, node.State().board))
            continue;
        weight *= result_.Strategy().ActionProbability({nodeId, hand}, childIndex);
    }
    return propagatedReach;
}

ReachCalculator::JointReachMasses ReachCalculator::PropagateChanceReach(
    const JointReachMasses& jointReachMasses,
    core::Card dealtCard,
    int legalOutcomeCount
) const
{
    if (legalOutcomeCount <= 0)
        throw std::runtime_error("Chance node has no legal outcomes");

    JointReachMasses propagatedMasses;
    propagatedMasses.reserve(jointReachMasses.size());
    for (const JointReach& jointReach : jointReachMasses)
    {
        if (core::Contains(jointReach.player0Hand, dealtCard) || core::Contains(jointReach.player1Hand, dealtCard))
            continue;
        propagatedMasses.push_back({
            jointReach.player0Hand,
            jointReach.player1Hand,
            jointReach.jointReachMass / legalOutcomeCount,
        });
    }
    return propagatedMasses;
}
} // namespace solver::analysis
