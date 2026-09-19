#pragma once

#include "core/Card.h"
#include "game/Identifiers.h"
#include <array>
#include <optional>
#include <vector>

namespace solver::analysis
{
struct HandEquity
{
    core::HoleCards cards;
    float ownReachWeight;
    std::optional<float> equity;
};

struct PlayerEquity
{
    std::optional<float> equity;
    std::vector<HandEquity> hands;
};

struct EquityReport
{
    game::NodeId nodeId;
    std::array<PlayerEquity, 2> players;
};
} // namespace solver::analysis
