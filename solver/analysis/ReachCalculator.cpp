#include "analysis/ReachCalculator.h"
#include "game/CompiledGame.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace solver::analysis
{
namespace
{
// Pairs are visited only while deriving reports; the path retains own reach alone.
// Returning false stops early once a support query has its answer.
template<typename Visitor>
void VisitPairs(const ReachCalculator::PlayerOwnReachWeights& own, const core::Board& board, Visitor visit)
{
    for (const auto& [first, firstWeight] : own.player0)
    {
        if (firstWeight <= 0.0f || core::Overlaps(first, board))
            continue;
        for (const auto& [second, secondWeight] : own.player1)
        {
            if (secondWeight <= 0.0f || core::Overlaps(second, board) || core::Overlaps(first, second))
                continue;
            const float weight = firstWeight * secondWeight;
            if (weight > 0.0f && !visit(first, second, weight))
                return;
        }
    }
}
} // namespace

ReachCalculator::ReachCalculator(const engine::SolveResult& result) : result_(result)
{
    const engine::SolveProblem& problem = result_.Problem();
    NodeReach root{{problem.ranges.For(core::PlayerId::Player0()).Entries(), problem.ranges.For(core::PlayerId::Player1()).Entries()}};
    VisitPairs(
        root.ownReachWeights,
        problem.game->Spec().initialBoard,
        [&](core::HoleCards, core::HoleCards, float weight)
        {
            rootMass_ += weight;
            return true;
        }
    );
    if (rootMass_ <= 0.0f)
        throw std::runtime_error("No valid private hand pairs for reach calculation");
    path_.push_back({problem.game->Root(), std::move(root)});
}

const ReachCalculator::NodeReach& ReachCalculator::ReachFor(game::NodeId nodeId)
{
    if (path_.back().node == nodeId)
        return path_.back().reach;

    const game::CompiledGame& game = *result_.Problem().game;
    std::vector<game::GameNode> nodes;
    for (auto node = game.GetNode(nodeId);; node = game.GetNode(node.Parent()->node))
    {
        nodes.push_back(node);
        if (!node.Parent())
            break;
    }
    std::reverse(nodes.begin(), nodes.end());
    std::size_t shared = 0;
    while (shared < path_.size() && shared < nodes.size() && path_[shared].node == nodes[shared].Id())
        ++shared;
    // Release abandoned branches before allocating the new path's reach tables.
    path_.erase(path_.begin() + shared, path_.end());
    for (std::size_t depth = shared; depth < nodes.size(); ++depth)
    {
        const auto& parent = nodes[depth - 1];
        const auto edge = nodes[depth].Parent()->edgeIndex;
        const auto& parentReach = path_.back().reach;
        NodeReach reach;
        reach.chanceProbability = parentReach.chanceProbability;
        if (parent.Kind() == game::NodeKind::Chance)
        {
            reach.chanceProbability /= 52 - parent.State().board.CardCount() - 4;
            reach.ownReachWeights = parentReach.ownReachWeights;
        }
        else
        {
            reach.ownReachWeights = PropagateOwnReach(parent.Id(), edge, parentReach.ownReachWeights);
        }
        path_.push_back({nodes[depth].Id(), std::move(reach)});
    }
    return path_.back().reach;
}

ReachCalculator::HandWeights ReachCalculator::BuildMarginalReachMasses(
    const NodeReach& reach,
    const core::Board& board,
    core::PlayerId player
) const
{
    HandWeights masses;
    VisitPairs(
        reach.ownReachWeights,
        board,
        [&](core::HoleCards first, core::HoleCards second, float weight)
        {
            masses[player == core::PlayerId::Player0() ? first : second] += (weight / rootMass_) * reach.chanceProbability;
            return true;
        }
    );
    return masses;
}

std::optional<std::uint64_t> ReachCalculator::CommonBlockers(const NodeReach& reach, const core::Board& board) const
{
    std::optional<std::uint64_t> common;
    VisitPairs(
        reach.ownReachWeights,
        board,
        [&](core::HoleCards first, core::HoleCards second, float weight)
        {
            if ((weight / rootMass_) * reach.chanceProbability <= 0.0f)
                return true;
            std::uint64_t blocked = 0;
            for (const auto hand : {first, second})
                for (const auto card : hand.Cards())
                    blocked |= std::uint64_t{1} << card.Index();
            common = common ? *common & blocked : blocked;
            return *common != 0;
        }
    );
    return common;
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

} // namespace solver::analysis
