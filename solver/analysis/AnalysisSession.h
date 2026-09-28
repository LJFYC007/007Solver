#pragma once

#include "analysis/NodeReport.h"
#include "analysis/EquityReport.h"
#include "analysis/ReachCalculator.h"
#include "engine/SolveResult.h"
#include "engine/StrategyEvaluator.h"
#include <map>
#include <memory>
#include <vector>

namespace solver::engine
{
struct HandTraversalData;
} // namespace solver::engine

namespace solver::analysis
{
class AnalysisSession
{
public:
    explicit AnalysisSession(engine::SolveResult result);
    AnalysisSession(const AnalysisSession&) = delete;
    AnalysisSession& operator=(const AnalysisSession&) = delete;
    AnalysisSession(AnalysisSession&&) = delete;
    AnalysisSession& operator=(AnalysisSession&&) = delete;

    game::NodeId RootNode() const { return result_.Problem().game->Root(); }
    // Decision reports include node and action EVs for hands with joint reach.
    NodeReport QueryNode(game::NodeId nodeId);
    EquityReport QueryEquity(game::NodeId nodeId);
    // The waiting player's node EVs at a decision, for hands with joint reach.
    OpponentEvReport QueryOpponentEv(game::NodeId nodeId);

private:
    engine::SolveResult result_;
    ReachCalculator reachCalculator_;
    // The last evaluated decision's traversal tables, since its opponent EV query usually follows
    // its node query. They stay until another decision is evaluated.
    std::shared_ptr<const engine::HandTraversalData> traversal_;

    std::vector<HandReport> BuildHandReports(
        game::NodeId nodeId,
        const core::Range& range,
        const ReachCalculator::HandWeights& marginalReachMasses,
        const ReachCalculator::HandWeights& ownReachWeights,
        core::PlayerId player
    ) const;
    engine::HandTraversal TraversalAt(game::NodeId nodeId);
    // Node strategy EVs of the player's hands with joint reach; the evaluator also omits hands whose
    // opponent mass has no value scale (see ValueScale). Other hands' EVs must stay null, not zero.
    std::map<core::HoleCards, engine::NodeStrategyValue> JointReachEvs(
        game::NodeId nodeId,
        core::PlayerId player,
        const ReachCalculator::HandWeights& marginalReachMasses
    );
};
} // namespace solver::analysis
