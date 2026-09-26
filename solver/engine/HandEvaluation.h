#pragma once

#include "engine/HandBoardData.h"

namespace solver::engine
{
// Conditional-value denominators use direct positive summation, without blocker subtraction.
std::vector<float> CompatibleHandMasses(const HandBoardData& data, std::size_t player, const float* opponentReach);

// Terminal values scale a hand's opponent-weighted payoff by the reciprocal of its compatible
// opponent mass, zero without any or when the reciprocal would overflow; both devices multiply
// by this same float.
inline float ValueScale(float mass)
{
    return mass > 0x1p-128f ? 1.0f / mass : 0.0f;
}
std::vector<float> ValueScales(const std::vector<float>& masses);

// Opponent hands overlapping boardMask must carry zero reach; chance propagation and
// the root tables guarantee that, so every holder list is summed without masking.
void EvaluateFoldHands(
    const HandBoardData& data,
    std::size_t player,
    std::uint64_t boardMask,
    const float* opponentReach,
    const float* scales,
    float utility,
    float* values
);

// Payoffs are win/tie/loss for this player. Rank rows exclude public-card blockers;
// opponentReach and scales (see ValueScale) use the board table's original hand indices.
void EvaluateShowdownHands(
    const HandBoardData& data,
    std::size_t player,
    const std::array<HandBoardData::RankOrder, 2>& ranks,
    const float* opponentReach,
    const float* scales,
    const std::array<float, 3>& utilities,
    float* values
);
} // namespace solver::engine
