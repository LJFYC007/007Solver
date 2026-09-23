#include "analysis/AnalysisSession.h"
#include "engine/HandBoardData.h"
#include "engine/HandEvaluation.h"
#include <array>

namespace solver::analysis
{
EquityReport AnalysisSession::QueryEquity(game::NodeId nodeId)
{
    const auto& problem = result_.Problem();
    const auto board = problem.game->GetNode(nodeId).State().board;
    const auto& reach = reachCalculator_.ReachFor(nodeId);
    EquityReport report{nodeId};
    const auto& weights = reach.ownReachWeights;
    for (std::size_t player = 0; player < 2; ++player)
        for (const auto& [hand, weight] : weights[player])
            if (weight > 0.0f && !core::Overlaps(hand, board))
                report.players[player].hands.push_back({hand, weight, std::nullopt});
    if (!reachCalculator_.CommonBlockers(reach, board))
        return report;

    // Query-local tables keep this input-thread calculation independent of the EV worker's cache.
    const engine::HandBoardData tables(problem, nodeId);
    const float runoutCount = board.CardCount() == 3 ? 990.0f : board.CardCount() == 4 ? 44.0f : 1.0f;
    std::array<std::vector<float>, 2> ownReach;
    for (std::size_t player = 0; player < 2; ++player)
        for (const auto& hand : tables.hands[player])
            ownReach[player].push_back(weights[player].at(hand.cards));

    for (std::size_t player = 0; player < 2; ++player)
    {
        const auto divisors = engine::CompatibleHandMasses(tables, player, ownReach[1 - player].data());
        std::vector<float> equities(tables.hands[player].size(), 0.0f), values(equities.size());
        for (const auto& ranks : tables.rankRows)
        {
            engine::EvaluateShowdownHands(
                tables, player, ranks, ownReach[1 - player].data(), divisors.data(), {1.0f, 0.5f, 0.0f}, values.data()
            );
            for (std::size_t hand = 0; hand < values.size(); ++hand)
                equities[hand] += values[hand];
        }
        float totalEquity = 0.0f, totalMass = 0.0f;
        std::size_t output = 0;
        for (std::size_t hand = 0; hand < equities.size(); ++hand)
        {
            if (ownReach[player][hand] <= 0.0f)
                continue;
            auto& entry = report.players[player].hands[output++];
            if (divisors[hand] > 0.0f)
                entry.equity = equities[hand] / runoutCount;
            const float mass = ownReach[player][hand] * divisors[hand];
            totalEquity += mass * (equities[hand] / runoutCount);
            totalMass += mass;
        }
        if (totalMass > 0.0f)
            report.players[player].equity = totalEquity / totalMass;
    }
    return report;
}
} // namespace solver::analysis
