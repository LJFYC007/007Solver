#include "engine/HandEvaluation.h"
#include <algorithm>
#include <array>

namespace solver::engine
{

std::vector<float> CompatibleHandMasses(const HandBoardData& data, std::size_t player, const float* opponentReach)
{
    std::vector<float> masses(data.hands[player].size(), 0.0f);
    for (std::size_t hand = 0; hand < data.hands[player].size(); ++hand)
        for (std::size_t other = 0; other < data.hands[1 - player].size(); ++other)
            if (!(data.hands[player][hand].mask & data.hands[1 - player][other].mask))
                masses[hand] += opponentReach[other];
    return masses;
}

std::vector<float> ValueScales(const std::vector<float>& masses)
{
    std::vector<float> scales;
    scales.reserve(masses.size());
    for (const float mass : masses)
        scales.push_back(ValueScale(mass));
    return scales;
}

void EvaluateFoldHands(
    const HandBoardData& data,
    std::size_t player,
    std::uint64_t boardMask,
    const float* opponentReach,
    const float* scales,
    float utility,
    float* values
)
{
    const auto& mine = data.hands[player];
    const auto& holders = data.holders[1 - player];
    // Board-blocked opponent hands carry zero reach, so every holder list adds up unchanged.
    float total = 0.0f;
    for (std::size_t other = 0; other < data.hands[1 - player].size(); ++other)
        total += opponentReach[other];
    std::array<float, 52> cards;
    for (int card = 0; card < 52; ++card)
    {
        float mass = 0.0f;
        for (auto entry = holders.cardOffsets[card]; entry < holders.cardOffsets[card + 1]; ++entry)
            mass += opponentReach[holders.cardLists[entry]];
        cards[card] = mass;
    }
    std::array<float, HandBoardData::kMaxHands> masses;
    for (std::size_t index = 0; index < mine.size(); ++index)
    {
        const auto& hand = mine[index];
        const float identical = hand.matchingOpponent < 0 ? 0.0f : opponentReach[hand.matchingOpponent];
        masses[index] = (hand.mask & boardMask) ? 0.0f : total - cards[hand.cardIndices[0]] - cards[hand.cardIndices[1]] + identical;
    }
    for (std::size_t index = 0; index < mine.size(); ++index)
        values[index] = scales[index] > 0.0f ? (masses[index] * scales[index]) * utility : 0.0f;
}

// Prefix sums over the opponent's rank order give each hand's strictly weaker mass in one
// lookup and its strictly stronger mass as the total less another; per-card runs remove the
// holders of its own cards.
void EvaluateShowdownHands(
    const HandBoardData& data,
    std::size_t player,
    const std::array<HandBoardData::RankOrder, 2>& ranks,
    const float* opponentReach,
    const float* scales,
    const std::array<float, 3>& utilities,
    float* values
)
{
    const auto& mine = ranks[player];
    const auto& opponent = ranks[1 - player];
    const auto myCount = mine.hands.size(), opponentCount = opponent.hands.size(), count = data.hands[player].size();
    const float tie = utilities[1], winDelta = utilities[0] - tie, lossDelta = utilities[2] - tie;
    std::array<float, HandBoardData::kMaxHands + 1> forward;
    forward[0] = 0.0f;
    for (std::size_t i = 0; i < opponentCount; ++i)
        forward[i + 1] = forward[i] + opponentReach[opponent.hands[i]];
    // Each card's run starts with a zero entry at cardOffsets[card] + card.
    std::array<float, 2 * HandBoardData::kMaxHands + 52> runs;
    for (int card = 0; card < 52; ++card)
    {
        const std::size_t begin = opponent.holders.cardOffsets[card], end = opponent.holders.cardOffsets[card + 1];
        float* run = runs.data() + begin + card;
        run[0] = 0.0f;
        for (auto entry = begin; entry < end; ++entry)
            run[entry - begin + 1] = run[entry - begin] + opponentReach[opponent.holders.cardLists[entry]];
    }
    std::array<float, HandBoardData::kMaxHands> wins, losses, masses;
    std::fill_n(wins.data(), count, 0.0f);
    std::fill_n(losses.data(), count, 0.0f);
    std::fill_n(masses.data(), count, 0.0f);
    const bool needMass = tie != 0.0f;
    for (std::size_t i = 0; i < myCount; ++i)
    {
        const auto hand = mine.hands[i];
        const auto blockers = mine.blockers[i];
        float blocked = 0.0f, blockedWins = 0.0f, blockedLosses = 0.0f;
        for (int c = 0; c < 2; ++c)
        {
            const auto span = c ? mine.runs1[i] : mine.runs0[i];
            const float* run = runs.data() + (span & 0xffffu);
            const std::size_t length = span >> 16;
            const std::size_t below = (blockers >> (16 * c)) & 0xffu, through = (blockers >> (16 * c + 8)) & 0xffu;
            blocked += run[length];
            blockedWins += run[below];
            // A card's list has at most 51 entries, so this difference loses only ulps of that card's mass.
            blockedLosses += run[length] - run[through];
        }
        wins[hand] = forward[mine.lowerBounds[i]] - blockedWins;
        losses[hand] = (forward[opponentCount] - forward[mine.upperBounds[i]]) - blockedLosses;
        if (needMass)
        {
            const int matching = data.hands[player][hand].matchingOpponent;
            masses[hand] = forward[opponentCount] - blocked + (matching < 0 ? 0.0f : opponentReach[matching]);
        }
    }
    for (std::size_t hand = 0; hand < count; ++hand)
        values[hand] = scales[hand] > 0.0f ? (tie * masses[hand] + winDelta * wins[hand] + lossDelta * losses[hand]) * scales[hand] : 0.0f;
}
} // namespace solver::engine
