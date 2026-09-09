#pragma once

#include "core/Range.h"
#include "game/GameSpec.h"
#include <string>

namespace solver::io
{
struct Scenario
{
    game::GameSpec game;
    core::RangeSet ranges;
    int iterations;
    std::string algorithm = "escfr";
};

Scenario LoadScenario(const std::string& jsonPath);
} // namespace solver::io
