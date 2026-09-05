#pragma once

#include "core/Card.h"
#include "core/PokerTypes.h"
#include <array>
#include <map>
#include <utility>
#include <vector>

namespace solver::core
{
class Range
{
public:
    using Table = std::map<HoleCards, float>;

    Range() = default;
    explicit Range(Table exactComboWeights);
    explicit Range(const std::vector<std::pair<HoleCards, float>>& exactComboWeights);

    float GetWeight(HoleCards hand) const;
    const Table& Entries() const { return handWeights_; }

private:
    Table handWeights_;
};

class RangeSet
{
public:
    RangeSet(Range player0, Range player1);

    const Range& For(PlayerId player) const { return ranges_[player.Index()]; }

private:
    std::array<Range, 2> ranges_;
};
} // namespace solver::core
