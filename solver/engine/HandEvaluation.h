#pragma once

#include "engine/HandBoardData.h"

namespace solver::engine
{
// Conditional-value denominators use direct positive summation, without blocker subtraction.
std::vector<float> CompatibleHandMasses(const HandBoardData& data, std::size_t player, const float* opponentReach);

void EvaluateFoldHands(
    const HandBoardData& data,
    std::size_t player,
    std::uint64_t boardMask,
    const float* opponentReach,
    const float* divisors,
    float utility,
    float* values
);

// Payoffs are win/tie/loss for this player. Rank rows exclude public-card blockers;
// opponentReach and divisors use the board table's original hand indices.
void EvaluateShowdownHands(
    const HandBoardData& data,
    std::size_t player,
    const std::array<HandBoardData::RankOrder, 2>& ranks,
    const float* opponentReach,
    const float* divisors,
    const std::array<float, 3>& utilities,
    float* values
);
} // namespace solver::engine
