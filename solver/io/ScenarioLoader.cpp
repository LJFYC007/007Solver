#include "io/ScenarioLoader.h"
#include "core/Card.h"
#include "core/Chips.h"
#include "io/RangeNotationParser.h"
#include <array>
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
using Json = nlohmann::basic_json<std::map, std::vector, std::string, bool, std::int64_t, std::uint64_t, float>;

core::Chips ParseChips(const Json& value, const std::string& field)
{
    const float chips = value.get<float>();
    if (!std::isfinite(chips) || chips < 0.0f)
        throw std::runtime_error(field + " must be a finite non-negative number");

    const float raw = chips * core::Chips::kUnitsPerChip;
    if (raw >= static_cast<float>(std::numeric_limits<std::int32_t>::max()))
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

std::vector<std::int64_t> ParsePercentages(const Json& values, const std::string& field)
{
    if (!values.is_array())
        throw std::runtime_error(field + " must be an array of positive percentages");
    std::vector<std::int64_t> percentages;
    for (const auto& value : values)
    {
        if (!value.is_number())
            throw std::runtime_error(field + " must contain numbers");
        const float percent = value.get<float>();
        const float scaled = percent * 100.0f;
        const float whole = std::floor(percent);
        const float hundredths = std::round((percent - whole) * 100.0f);
        // Round the fractional part before adding it to the integer percentage.
        if (!std::isfinite(percent) || scaled < 1.0f || scaled >= static_cast<float>(std::numeric_limits<std::int32_t>::max()) ||
            percent != whole + hundredths / 100.0f)
            throw std::runtime_error(field + " must contain positive percentages with at most two decimal places");
        percentages.push_back(static_cast<std::int64_t>(whole) * 100 + static_cast<std::int64_t>(hundredths));
    }
    return percentages;
}

game::BettingAbstraction ParseBettingTree(const Json& json)
{
    const auto& tree = json.at("bettingTree");
    if (!tree.is_object())
        throw std::runtime_error("bettingTree must be an object");
    std::array<game::StreetBettingSizes, 3> streets;
    const std::array<std::string, 3> streetNames{"flop", "turn", "river"};
    for (std::size_t i = 0; i < streetNames.size(); ++i)
    {
        const auto& name = streetNames[i];
        const auto& street = tree.at(name);
        if (!street.is_object())
            throw std::runtime_error("bettingTree." + name + " must be an object");
        auto& sizes = streets[i];
        for (const auto percent : ParsePercentages(street.at("bet"), "bettingTree." + name + ".bet"))
            sizes.betSizes.push_back({game::BetSizeKind::PotFractionOfCurrentPot, percent, 10000, {}});
        for (const auto percent : ParsePercentages(street.at("raise"), "bettingTree." + name + ".raise"))
            sizes.raiseSizes.push_back({game::RaiseSizeKind::PotFractionOfPotAfterCallAsRaiseBy, percent, 10000, {}});
    }
    const auto& maxRaises = tree.at("maxRaises");
    if (!maxRaises.is_number_unsigned() || maxRaises.get<std::uint64_t>() > 2)
        throw std::runtime_error("bettingTree.maxRaises must be 0, 1 or 2 (excluding the opening bet)");
    const auto& allInSpr = tree.at("allInSpr");
    if (!allInSpr.is_number() || !std::isfinite(allInSpr.get<float>()) || allInSpr.get<float>() < 0.0f)
        throw std::runtime_error("bettingTree.allInSpr must be a finite non-negative number");
    return {std::move(streets), maxRaises.get<std::uint32_t>(), allInSpr.get<float>()};
}
} // namespace

Scenario LoadScenario(const std::string& jsonPath)
{
    std::ifstream file(jsonPath);
    if (!file.is_open())
        throw std::runtime_error("Cannot open scenario file: " + jsonPath);

    return ReadScenario(file);
}

Scenario ReadScenario(std::istream& input)
{
    Json json;
    input >> json;

    const Json& iterationCount = json.at("iterations");
    if (!iterationCount.is_number_unsigned() || iterationCount.get<std::uint64_t>() == 0 ||
        iterationCount.get<std::uint64_t>() > std::numeric_limits<int>::max())
        throw std::runtime_error("Solve iterations must be an integer between 1 and INT_MAX");
    const int iterations = iterationCount.get<int>();
    const float accuracyPercent = json.value("accuracyPercent", 0.01f);
    if (!std::isfinite(accuracyPercent) || accuracyPercent <= 0.0f)
        throw std::runtime_error("Accuracy must be a finite positive percentage of the initial pot");
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
            ParseBettingTree(json),
        },
        core::RangeSet(LoadRange(ranges, heroPosition), LoadRange(ranges, villainPosition)),
        iterations,
        accuracyPercent,
    };
}
} // namespace solver::io
