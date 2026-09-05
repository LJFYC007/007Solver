#include "core/Evaluator.h"
#include "SevenEval.h"
#include <array>
#include <cstdint>
#include <stdexcept>

namespace solver::core
{
int EvaluateHoldem(HoleCards hand, const Board& board)
{
    if (board.CardCount() != 5)
        throw std::runtime_error("Hold'em showdown evaluation requires five public cards");
    if (Overlaps(hand, board))
        throw std::runtime_error("Private hand overlaps the public board");

    std::array<std::uint8_t, 7> cards{};
    const std::array<Card, 2> privateCards = hand.Cards();
    cards[0] = static_cast<std::uint8_t>(privateCards[0].Index());
    cards[1] = static_cast<std::uint8_t>(privateCards[1].Index());
    for (int position = 0; position < 5; ++position)
        cards[position + 2] = static_cast<std::uint8_t>(board.CardAt(position).Index());

    return SevenEval::GetRank(cards[0], cards[1], cards[2], cards[3], cards[4], cards[5], cards[6]);
}
} // namespace solver::core
