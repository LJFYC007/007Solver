#include "io/RangeNotationParser.h"
#include "core/Card.h"
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace solver::io
{
core::Range ParseRangeNotation(const std::vector<std::pair<std::string, float>>& weightedHandClasses)
{
    const std::string ranks = "AKQJT98765432";
    constexpr char suits[] = {'s', 'h', 'd', 'c'};
    std::vector<std::pair<core::HoleCards, float>> exactCombos;

    for (const auto& [handClass, weight] : weightedHandClasses)
    {
        if (weight < 0.0f || weight > 1.0f)
            throw std::runtime_error("Range weight must be between 0 and 1 for hand class: " + handClass);
        if (handClass.size() != 2 && handClass.size() != 3)
            throw std::runtime_error("Invalid range hand class: " + handClass);

        const std::size_t firstRank = ranks.find(handClass[0]);
        const std::size_t secondRank = ranks.find(handClass[1]);
        const bool isPair = handClass.size() == 2 && handClass[0] == handClass[1];
        const bool isSuitedOrOffsuit =
            handClass.size() == 3 && handClass[0] != handClass[1] && (handClass[2] == 's' || handClass[2] == 'o') && firstRank < secondRank;
        if (firstRank == std::string::npos || secondRank == std::string::npos || (!isPair && !isSuitedOrOffsuit))
            throw std::runtime_error("Invalid range hand class: " + handClass);

        for (std::size_t firstSuit = 0; firstSuit < 4; ++firstSuit)
        {
            for (std::size_t secondSuit = 0; secondSuit < 4; ++secondSuit)
            {
                if ((isPair && firstSuit >= secondSuit) || (!isPair && (handClass[2] == 's') != (firstSuit == secondSuit)))
                    continue;

                std::string firstCard{handClass[0], suits[firstSuit]};
                std::string secondCard{handClass[1], suits[secondSuit]};
                exactCombos.emplace_back(core::HoleCards(core::ParseCard(firstCard), core::ParseCard(secondCard)), weight);
            }
        }
    }
    return core::Range(exactCombos);
}
} // namespace solver::io
