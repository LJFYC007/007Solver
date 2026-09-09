#include "io/ScenarioLoader.h"
#include "core/Card.h"
#include "core/Chips.h"
#include "io/RangeNotationParser.h"
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace solver::io
{
namespace
{
using Json = nlohmann::json;

core::Chips ParseChips(const Json& value, const std::string& field)
{
    const double chips = value.get<double>();
    if (!std::isfinite(chips) || chips < 0.0)
        throw std::runtime_error(field + " must be a finite non-negative number");

    const double raw = chips * static_cast<double>(core::Chips::kUnitsPerChip);
    if (raw > static_cast<double>(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error(field + " is too large");
    return core::Chips::FromRaw(static_cast<std::int32_t>(std::llround(raw)));
}

core::Range LoadRange(const Json& ranges, const std::string& position)
{
    const auto positionIt = ranges.find(position);
    if (positionIt == ranges.end())
        throw std::runtime_error("Position not found in ranges: " + position);
    if (!positionIt->is_object())
        throw std::runtime_error("Range must be a JSON object for position: " + position);

    std::vector<std::pair<std::string, float>> weightedHandClasses;
    weightedHandClasses.reserve(positionIt->size());
    for (const auto& [handClass, weight] : positionIt->items())
        weightedHandClasses.emplace_back(handClass, weight.get<float>());
    return ParseRangeNotation(weightedHandClasses);
}
} // namespace

Scenario LoadScenario(const std::string& jsonPath)
{
    std::ifstream file(jsonPath);
    if (!file.is_open())
        throw std::runtime_error("Cannot open scenario file: " + jsonPath);

    Json json;
    file >> json;

    const Json& iterationCount = json.at("iterations");
    if (!iterationCount.is_number_unsigned() || iterationCount.get<std::uint64_t>() == 0 ||
        iterationCount.get<std::uint64_t>() > std::numeric_limits<int>::max())
        throw std::runtime_error("Solve iterations must be an integer between 1 and INT_MAX");
    const int iterations = iterationCount.get<int>();
    if (json.value("algorithm", std::string("dcfr")) != "dcfr")
        throw std::runtime_error("Solve algorithm must be dcfr");

    const std::string heroPosition = json.at("heroPosition").get<std::string>();
    const std::string villainPosition = json.at("villainPosition").get<std::string>();
    const Json& ranges = json.at("ranges");
    return {
        {
            core::ParseBoard(json.at("board").get<std::string>(), 3),
            core::Street::Flop,
            ParseChips(json.at("initialPot"), "initialPot"),
            {
                ParseChips(json.at("heroStack"), "heroStack"),
                ParseChips(json.at("villainStack"), "villainStack"),
            },
            json.at("heroActsFirst").get<bool>() ? core::PlayerId::Player0() : core::PlayerId::Player1(),
            game::BettingAbstraction::Default(),
        },
        core::RangeSet(LoadRange(ranges, heroPosition), LoadRange(ranges, villainPosition)),
        iterations,
    };
}
} // namespace solver::io
