#include "engine/HandBoardData.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
namespace
{
// Lists each card's holders among hands[handAt(i)] for i < count, in visiting order.
template<typename HandAt>
void ListHolders(HandBoardData::CardLists& lists, const std::vector<HandBoardData::Hand>& hands, std::size_t count, HandAt handAt)
{
    std::array<std::uint16_t, 52> counts{};
    for (std::size_t i = 0; i < count; ++i)
        for (const auto card : hands[handAt(i)].cardIndices)
            ++counts[card];
    for (int card = 0; card < 52; ++card)
        lists.cardOffsets[card + 1] = static_cast<std::uint16_t>(lists.cardOffsets[card] + counts[card]);
    lists.cardLists.resize(2 * count);
    std::array<std::uint16_t, 52> next{};
    std::copy(lists.cardOffsets.begin(), lists.cardOffsets.begin() + 52, next.begin());
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto hand = handAt(i);
        for (const auto card : hands[hand].cardIndices)
            lists.cardLists[next[card]++] = static_cast<std::uint16_t>(hand);
    }
}
} // namespace

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
        cardFactors[player].assign(52 * hands[player].size(), 1.0f);
        for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
        {
            hands[player][hand].weight /= totalWeight;
            handMasks[player][hand] = hands[player][hand].mask;
            for (const auto card : hands[player][hand].cardIndices)
                cardFactors[player][card * hands[player].size() + hand] = 0.0f;
        }
    }

    for (std::size_t player = 0; player < 2; ++player)
        ListHolders(holders[player], hands[player], hands[player].size(), [](std::size_t hand) { return hand; });

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
        const auto key = core::CardPairIndex(board.CardAt(3), board.CardAt(4));
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
            for (const auto& [rank, hand] : ranked)
            {
                order.ranks.push_back(rank);
                order.hands.push_back(hand);
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
            // Ranked visiting order keeps each card's holders in rank order.
            ListHolders(order.holders, hands[player], order.hands.size(), [&](std::size_t ranked) { return order.hands[ranked]; });
        }
        for (std::size_t player = 0; player < 2; ++player)
        {
            auto& order = rankRows.back()[player];
            const auto& opponent = rankRows.back()[1 - player];
            std::vector<std::uint16_t> rankByHand(hands[1 - player].size());
            for (std::size_t ranked = 0; ranked < opponent.hands.size(); ++ranked)
                rankByHand[opponent.hands[ranked]] = opponent.ranks[ranked];
            order.blockers.reserve(order.hands.size());
            order.runs0.reserve(order.hands.size());
            order.runs1.reserve(order.hands.size());
            const auto holderRank = [&](std::uint16_t hand) { return rankByHand[hand]; };
            for (std::size_t ranked = 0; ranked < order.hands.size(); ++ranked)
            {
                const auto rank = order.ranks[ranked];
                std::uint32_t packed = 0;
                for (int c = 0; c < 2; ++c)
                {
                    const auto card = hands[player][order.hands[ranked]].cardIndices[c];
                    const auto begin = opponent.holders.cardLists.begin() + opponent.holders.cardOffsets[card];
                    const auto end = opponent.holders.cardLists.begin() + opponent.holders.cardOffsets[card + 1];
                    // Holder lists are rank-sorted, so the boundaries are binary searches.
                    const auto below =
                        std::lower_bound(begin, end, rank, [&](std::uint16_t hand, std::uint16_t r) { return holderRank(hand) < r; });
                    const auto through =
                        std::upper_bound(begin, end, rank, [&](std::uint16_t r, std::uint16_t hand) { return r < holderRank(hand); });
                    packed |= (std::uint32_t(below - begin) | (std::uint32_t(through - begin) << 8)) << (16 * c);
                    (c ? order.runs1 : order.runs0)
                        .push_back(std::uint32_t(opponent.holders.cardOffsets[card] + card) | (std::uint32_t(end - begin) << 16));
                }
                order.blockers.push_back(packed);
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
