#include "engine/HandBoardData.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
HandBoardData::HandBoardData(const SolveProblem& problem, game::NodeId root)
    : game(problem.game)
    , board(game ? game->GetNode(root).State().board : throw std::invalid_argument("Hand tables require a compiled game"))
{
    const game::CompiledGame& game = *this->game;
    for (std::uint8_t player = 0; player < 2; ++player)
    {
        float totalWeight = 0.0f;
        for (const auto& [cards, weight] : problem.ranges.For(core::PlayerId(player)).Entries())
        {
            if (weight <= 0.0f || core::Overlaps(cards, this->board))
                continue;
            const auto pair = cards.Cards();
            const auto first = static_cast<std::uint8_t>(pair[0].Index());
            const auto second = static_cast<std::uint8_t>(pair[1].Index());
            hands[player].push_back({cards, weight, (std::uint64_t{1} << first) | (std::uint64_t{1} << second), {first, second}});
            totalWeight += weight;
        }
        // A constant scale per player's range leaves regret matching unchanged.
        handMasks[player].resize(hands[player].size());
        for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
        {
            hands[player][hand].weight /= totalWeight;
            handMasks[player][hand] = hands[player][hand].mask;
        }
    }

    bool hasLegalPair = false;
    for (std::size_t first = 0; first < hands[0].size(); ++first)
    {
        for (std::size_t second = 0; second < hands[1].size(); ++second)
        {
            if (!(hands[0][first].mask & hands[1][second].mask))
            {
                hasLegalPair = true;
                hands[0][first].opponentMass += hands[1][second].weight;
                hands[1][second].opponentMass += hands[0][first].weight;
            }
            if (hands[0][first].cards == hands[1][second].cards)
            {
                hands[0][first].matchingOpponent = static_cast<int>(second);
                hands[1][second].matchingOpponent = static_cast<int>(first);
            }
        }
    }
    if (!hasLegalPair)
        throw std::runtime_error("No valid private hand pairs after applying range weights and blockers");

    rowsByRunout.fill(-1);
    // Rank rows are shared by all betting histories and reversed turn/river runouts.
    const auto addRanks = [&](const core::Board& board)
    {
        const int first = board.CardAt(3).Index();
        const int second = board.CardAt(4).Index();
        const int high = std::max(first, second);
        const int key = high * (high - 1) / 2 + std::min(first, second);
        if (rowsByRunout[key] >= 0)
            return;
        rowsByRunout[key] = static_cast<int>(rankRows.size());
        rankRows.emplace_back();
        for (std::size_t player = 0; player < 2; ++player)
        {
            std::vector<std::pair<std::uint16_t, std::uint16_t>> ranked;
            ranked.reserve(hands[player].size());
            for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
                if (!core::Overlaps(hands[player][hand].cards, board))
                    ranked.emplace_back(
                        static_cast<std::uint16_t>(game.ShowdownRank(board, hands[player][hand].cards)), static_cast<std::uint16_t>(hand)
                    );
            std::sort(ranked.begin(), ranked.end());
            auto& order = rankRows.back()[player];
            order.ranks.reserve(ranked.size());
            order.hands.reserve(ranked.size());
            order.card0.reserve(ranked.size());
            order.card1.reserve(ranked.size());
            order.masks.reserve(ranked.size());
            for (const auto& [rank, hand] : ranked)
            {
                const Hand& source = hands[player][hand];
                order.ranks.push_back(rank);
                order.hands.push_back(hand);
                order.card0.push_back(source.cardIndices[0]);
                order.card1.push_back(source.cardIndices[1]);
                order.masks.push_back(source.mask);
            }
        }
        for (std::size_t player = 0; player < 2; ++player)
        {
            auto& order = rankRows.back()[player];
            const auto& opponent = rankRows.back()[1 - player].ranks;
            order.lowerBounds.reserve(order.ranks.size());
            order.upperBounds.reserve(order.ranks.size());
            for (const auto rank : order.ranks)
            {
                order.lowerBounds.push_back(
                    static_cast<std::uint16_t>(std::lower_bound(opponent.begin(), opponent.end(), rank) - opponent.begin())
                );
                order.upperBounds.push_back(
                    static_cast<std::uint16_t>(std::upper_bound(opponent.begin(), opponent.end(), rank) - opponent.begin())
                );
            }
        }
    };
    if (this->board.CardCount() == 5)
        addRanks(this->board);
    else
        for (int first = 0; first < 52; ++first)
        {
            if (core::Contains(this->board, core::Card(first)))
                continue;
            const auto board = this->board.Append(core::Card(first));
            if (board.CardCount() == 5)
                addRanks(board);
            else
                for (int second = 0; second < first; ++second)
                    if (!core::Contains(board, core::Card(second)))
                        addRanks(board.Append(core::Card(second)));
        }
}
} // namespace solver::engine
