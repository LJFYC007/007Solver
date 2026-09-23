#include "core/Range.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace solver::core
{
namespace
{
void ValidateWeight(float weight)
{
    if (!std::isfinite(weight) || weight < 0.0f || weight > 1.0f)
        throw std::runtime_error("Exact-combo range weight must be between 0 and 1");
}
} // namespace

Range::Range(const std::vector<std::pair<HoleCards, float>>& exactComboWeights)
{
    for (const auto& [hand, weight] : exactComboWeights)
    {
        ValidateWeight(weight);
        if (!handWeights_.emplace(hand, weight).second)
        {
            const std::array<Card, 2> cards = hand.Cards();
            throw std::runtime_error("Duplicate exact combo in range: " + FormatCard(cards[0]) + " " + FormatCard(cards[1]));
        }
    }
}

RangeSet::RangeSet(Range player0, Range player1) : ranges_{std::move(player0), std::move(player1)} {}
} // namespace solver::core
