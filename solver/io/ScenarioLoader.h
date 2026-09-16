#pragma once

#include "core/Range.h"
#include "game/GameSpec.h"
#include <istream>
#include <string>

namespace solver::io
{
struct Scenario
{
    game::GameSpec game;
    core::RangeSet ranges;
    int iterations;
    double accuracyPercent = 0.01;
};

Scenario LoadScenario(const std::string& jsonPath);
Scenario ReadScenario(std::istream& input);
} // namespace solver::io
