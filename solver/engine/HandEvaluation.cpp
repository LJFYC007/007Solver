#include "engine/HandEvaluation.h"
#include <algorithm>

namespace solver::engine
{
namespace
{
struct Mass
{
    float total = 0.0f;
    std::array<float, 52> cards{};

    void Add(float weight, std::uint8_t first, std::uint8_t second)
    {
        total += weight;
        cards[first] += weight;
        cards[second] += weight;
    }

    float Without(std::uint8_t first, std::uint8_t second) const { return total - cards[first] - cards[second]; }
};
} // namespace

std::vector<float> CompatibleHandMasses(const HandBoardData& data, std::size_t player, const float* opponentReach)
{
    std::vector<float> masses(data.hands[player].size(), 0.0f);
    for (std::size_t hand = 0; hand < data.hands[player].size(); ++hand)
        for (std::size_t other = 0; other < data.hands[1 - player].size(); ++other)
            if (!(data.hands[player][hand].mask & data.hands[1 - player][other].mask))
                masses[hand] += opponentReach[other];
    return masses;
}

void EvaluateFoldHands(
    const HandBoardData& data,
    std::size_t player,
    std::uint64_t boardMask,
    const float* opponentReach,
    const float* divisors,
    float utility,
    float* values
)
{
    const auto& mine = data.hands[player];
    const auto& opponent = data.hands[1 - player];
    Mass aggregate;
    for (std::size_t other = 0; other < opponent.size(); ++other)
        if (!(opponent[other].mask & boardMask))
            aggregate.Add(opponentReach[other], opponent[other].cardIndices[0], opponent[other].cardIndices[1]);
    for (std::size_t index = 0; index < mine.size(); ++index)
    {
        const auto& hand = mine[index];
        values[index] = 0.0f;
        if ((hand.mask & boardMask) || divisors[index] <= 0.0f)
            continue;
        const float identical = hand.matchingOpponent < 0 ? 0.0f : opponentReach[hand.matchingOpponent];
        const float mass = aggregate.Without(hand.cardIndices[0], hand.cardIndices[1]) + identical;
        values[index] = (mass / divisors[index]) * utility;
    }
}

void EvaluateShowdownHands(
    const HandBoardData& data,
    std::size_t player,
    const std::array<HandBoardData::RankOrder, 2>& ranks,
    const float* opponentReach,
    const float* divisors,
    const std::array<float, 3>& utilities,
    float* values
)
{
    const auto& mine = ranks[player];
    const auto& opponent = ranks[1 - player];
    const auto myCount = mine.hands.size(), opponentCount = opponent.hands.size();
    const float tie = utilities[1], winDelta = utilities[0] - tie, lossDelta = utilities[2] - tie;
    std::fill_n(values, data.hands[player].size(), 0.0f);
    // Zero-tie solver roots need only the strict win/loss sweeps.
    if (tie != 0.0f)
    {
        Mass aggregate;
        for (std::size_t other = 0; other < opponentCount; ++other)
            aggregate.Add(opponentReach[opponent.hands[other]], opponent.card0[other], opponent.card1[other]);
        for (std::size_t ranked = 0; ranked < myCount; ++ranked)
        {
            const auto hand = mine.hands[ranked];
            if (divisors[hand] <= 0.0f)
                continue;
            const int matching = data.hands[player][hand].matchingOpponent;
            const float mass = aggregate.Without(mine.card0[ranked], mine.card1[ranked]) + (matching < 0 ? 0.0f : opponentReach[matching]);
            values[hand] = (mass / divisors[hand]) * tie;
        }
    }
    Mass lower;
    std::size_t cursor = 0;
    for (std::size_t ranked = 0; ranked < myCount; ++ranked)
    {
        while (cursor < mine.lowerBounds[ranked])
        {
            lower.Add(opponentReach[opponent.hands[cursor]], opponent.card0[cursor], opponent.card1[cursor]);
            ++cursor;
        }
        const auto hand = mine.hands[ranked];
        if (divisors[hand] > 0.0f)
        {
            const float mass = lower.Without(mine.card0[ranked], mine.card1[ranked]);
            values[hand] += (mass / divisors[hand]) * winDelta;
        }
    }
    Mass upper;
    cursor = opponentCount;
    for (std::size_t ranked = myCount; ranked > 0;)
    {
        --ranked;
        while (cursor > mine.upperBounds[ranked])
        {
            --cursor;
            upper.Add(opponentReach[opponent.hands[cursor]], opponent.card0[cursor], opponent.card1[cursor]);
        }
        const auto hand = mine.hands[ranked];
        if (divisors[hand] > 0.0f)
        {
            const float mass = upper.Without(mine.card0[ranked], mine.card1[ranked]);
            values[hand] += (mass / divisors[hand]) * lossDelta;
        }
    }
}
} // namespace solver::engine
