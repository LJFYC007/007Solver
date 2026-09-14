#include "analysis/AnalysisSession.h"
#include <algorithm>
#include <array>
#include <map>

namespace solver::analysis
{
namespace
{
struct RankedHand
{
    std::size_t index;
    int rank;
};

struct Mass
{
    double total = 0.0;
    std::array<double, 52> cards{};

    void Add(const HandEquity& hand)
    {
        total += hand.ownReachWeight;
        for (core::Card card : hand.cards.Cards())
            cards[card.Index()] += hand.ownReachWeight;
    }

    double Without(core::HoleCards hand) const { return total - cards[hand.CardAt(0).Index()] - cards[hand.CardAt(1).Index()]; }
};
} // namespace

EquityReport AnalysisSession::QueryEquity(game::NodeId nodeId)
{
    const auto& game = *result_.Problem().game;
    const auto board = game.GetNode(nodeId).State().board;
    const auto& reach = reachCalculator_.ReachFor(nodeId);
    EquityReport report{nodeId};
    const std::array<const ReachCalculator::HandWeights*, 2> weights{&reach.ownReachWeights.player0, &reach.ownReachWeights.player1};
    std::array<std::map<core::HoleCards, std::size_t>, 2> indices;
    for (std::size_t player = 0; player < 2; ++player)
        for (const auto& [hand, weight] : *weights[player])
            if (weight > 0.0f && !core::Overlaps(hand, board))
            {
                indices[player].emplace(hand, report.players[player].hands.size());
                report.players[player].hands.push_back({hand, weight, std::nullopt});
            }
    if (reach.jointReachMasses.empty())
        return report;

    std::array<std::vector<double>, 2> wins, masses;
    for (std::size_t player = 0; player < 2; ++player)
    {
        wins[player].resize(report.players[player].hands.size());
        masses[player].resize(report.players[player].hands.size());
    }

    const auto evaluate = [&](const core::Board& river)
    {
        std::array<std::vector<RankedHand>, 2> ranks;
        std::array<Mass, 2> totals;
        for (std::size_t player = 0; player < 2; ++player)
        {
            const auto& hands = report.players[player].hands;
            for (std::size_t index = 0; index < hands.size(); ++index)
                if (!core::Overlaps(hands[index].cards, river))
                {
                    ranks[player].push_back({index, game.ShowdownRank(river, hands[index].cards)});
                    totals[player].Add(hands[index]);
                }
            std::sort(ranks[player].begin(), ranks[player].end(), [](auto a, auto b) { return a.rank < b.rank; });
        }
        for (std::size_t player = 0; player < 2; ++player)
        {
            const auto& opponents = report.players[1 - player].hands;
            const auto& opponentRanks = ranks[1 - player];
            Mass lower;
            std::size_t cursor = 0;
            for (std::size_t first = 0; first < ranks[player].size();)
            {
                const int rank = ranks[player][first].rank;
                while (cursor < opponentRanks.size() && opponentRanks[cursor].rank < rank)
                    lower.Add(opponents[opponentRanks[cursor++].index]);
                Mass tied;
                std::size_t end = cursor;
                while (end < opponentRanks.size() && opponentRanks[end].rank == rank)
                    tied.Add(opponents[opponentRanks[end++].index]);
                do
                {
                    const auto index = ranks[player][first++].index;
                    const auto hand = report.players[player].hands[index].cards;
                    const auto same = indices[1 - player].find(hand);
                    // An identical opponent hand was subtracted once for each shared card.
                    const double identical = same == indices[1 - player].end() ? 0.0 : opponents[same->second].ownReachWeight;
                    const auto compatible = [&](const Mass& source, std::size_t begin, std::size_t endIndex, double matching)
                    {
                        double value = source.Without(hand) + matching;
                        // Subtracting a dominant blocker can erase a small but positive
                        // compatible range. Match the traversal kernel's direct fallback.
                        if (value <= source.total * 1e-6)
                        {
                            value = 0.0;
                            for (std::size_t other = begin; other < endIndex; ++other)
                            {
                                const auto& opponent = opponents[opponentRanks[other].index];
                                if (!core::Overlaps(hand, opponent.cards))
                                    value += opponent.ownReachWeight;
                            }
                        }
                        return value;
                    };
                    const double mass = compatible(totals[1 - player], 0, opponentRanks.size(), identical);
                    const double tieMass = compatible(tied, cursor, end, identical);
                    masses[player][index] += mass;
                    wins[player][index] += compatible(lower, 0, cursor, 0.0) + 0.5 * tieMass;
                } while (first < ranks[player].size() && ranks[player][first].rank == rank);
            }
        }
    };

    // Every compatible private pair has the same number of legal runouts. Enumerating
    // public completions once and discarding blockers therefore preserves joint weights.
    if (board.CardCount() == 5)
        evaluate(board);
    else
        for (int first = 0; first < 52; ++first)
        {
            if (core::Contains(board, core::Card(first)))
                continue;
            const auto next = board.Append(core::Card(first));
            if (board.CardCount() == 4)
                evaluate(next);
            else
                for (int second = first + 1; second < 52; ++second)
                    if (!core::Contains(board, core::Card(second)))
                        evaluate(next.Append(core::Card(second)));
        }

    for (std::size_t player = 0; player < 2; ++player)
    {
        double totalWins = 0.0, totalMass = 0.0;
        for (std::size_t index = 0; index < report.players[player].hands.size(); ++index)
        {
            auto& hand = report.players[player].hands[index];
            if (masses[player][index] > 0.0)
                hand.equity = std::clamp(wins[player][index] / masses[player][index], 0.0, 1.0);
            totalWins += hand.ownReachWeight * wins[player][index];
            totalMass += hand.ownReachWeight * masses[player][index];
        }
        if (totalMass > 0.0)
            report.players[player].equity = std::clamp(totalWins / totalMass, 0.0, 1.0);
    }
    return report;
}
} // namespace solver::analysis
