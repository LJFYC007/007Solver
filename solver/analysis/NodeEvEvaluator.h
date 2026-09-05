#pragma once

#include "core/Card.h"
#include "engine/SolveResult.h"
#include "game/Identifiers.h"
#include <map>

namespace solver::analysis
{
float CalculateNodeHandEv(
    const engine::SolveResult& result,
    game::NodeId nodeId,
    core::HoleCards hand,
    core::PlayerId player,
    const std::map<core::HoleCards, float>& opponentReachMasses
);
} // namespace solver::analysis
