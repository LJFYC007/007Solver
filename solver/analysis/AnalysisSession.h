#pragma once

#include "analysis/NodeReport.h"
#include "analysis/ReachCalculator.h"
#include "engine/SolveResult.h"
#include <vector>

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
    NodeReport QueryNode(game::NodeId nodeId);

private:
    engine::SolveResult result_;
    ReachCalculator reachCalculator_;

    std::vector<HandReport> BuildHandReports(
        game::NodeId nodeId,
        const core::Range& range,
        const ReachCalculator::HandWeights& marginalReachMasses,
        const ReachCalculator::HandWeights& ownReachWeights,
        core::PlayerId player
    ) const;
};
} // namespace solver::analysis
