#include "core/Card.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace solver::core
{
namespace
{
int PackCards(const std::vector<Card>& cards, bool sortCards)
{
    if (cards.empty() || cards.size() > 5)
        throw std::runtime_error("Card encoding requires between 1 and 5 cards");

    std::uint64_t seenCards = 0;
    std::vector<Card> packedCards = cards;
    for (const Card card : packedCards)
    {
        const std::uint64_t cardMask = std::uint64_t{1} << card.Index();
        if ((seenCards & cardMask) != 0)
            throw std::runtime_error("Encoded cards contain duplicate cards");
        seenCards |= cardMask;
    }
    if (sortCards)
        std::sort(packedCards.begin(), packedCards.end());

    int result = 0;
    for (const Card card : packedCards)
        result = (result << 6) | (card.Index() + 1);
    return result;
}

int CardIndexAt(int packed, int position)
{
    return ((packed >> (position * 6)) & 0x3F) - 1;
}

std::vector<Card> ParseCards(const std::string& text)
{
    std::istringstream stream(text);
    std::string token;
    std::vector<Card> cards;
    while (stream >> token)
    {
        if (cards.size() == 5)
            throw std::runtime_error("Card string cannot contain more than 5 cards");
        cards.push_back(ParseCard(token));
    }
    if (cards.empty())
        throw std::runtime_error("Card string cannot be empty");
    return cards;
}
} // namespace

Card::Card(int index) : index_(index)
{
    if (index_ < 0 || index_ >= 52)
        throw std::out_of_range("Invalid card index");
}

HoleCards::HoleCards(Card first, Card second) : packed_(PackCards({first, second}, true)) {}

Card HoleCards::CardAt(int position) const
{
    if (position < 0 || position >= 2)
        throw std::out_of_range("Private card position must be 0 or 1");
    return Card(CardIndexAt(packed_, position));
}

std::array<Card, 2> HoleCards::Cards() const
{
    return {CardAt(1), CardAt(0)};
}

Board::Board(const std::vector<Card>& cards) : packed_(PackCards(cards, false)), cardCount_(static_cast<int>(cards.size())) {}

Board::Board(int packed, int cardCount) : packed_(packed), cardCount_(cardCount) {}

Card Board::CardAt(int dealOrderIndex) const
{
    if (dealOrderIndex < 0 || dealOrderIndex >= cardCount_)
        throw std::out_of_range("Public card position is outside the board");
    return Card(CardIndexAt(packed_, cardCount_ - 1 - dealOrderIndex));
}

Board Board::Append(Card card) const
{
    if (cardCount_ >= 5)
        throw std::runtime_error("Cannot append a sixth public card");
    if (Contains(*this, card))
        throw std::runtime_error("Public board contains duplicate cards");

    return Board((packed_ << 6) | (card.Index() + 1), cardCount_ + 1);
}

Card ParseCard(const std::string& text)
{
    if (text.length() != 2)
        throw std::runtime_error("Invalid card format: each card must be exactly 2 characters");

    int rank = 0;
    switch (text[0])
    {
    case '2':
        rank = 12;
        break;
    case '3':
        rank = 11;
        break;
    case '4':
        rank = 10;
        break;
    case '5':
        rank = 9;
        break;
    case '6':
        rank = 8;
        break;
    case '7':
        rank = 7;
        break;
    case '8':
        rank = 6;
        break;
    case '9':
        rank = 5;
        break;
    case 'T':
    case 't':
        rank = 4;
        break;
    case 'J':
    case 'j':
        rank = 3;
        break;
    case 'Q':
    case 'q':
        rank = 2;
        break;
    case 'K':
    case 'k':
        rank = 1;
        break;
    case 'A':
    case 'a':
        rank = 0;
        break;
    default:
        throw std::runtime_error("Invalid rank character");
    }

    int suit = 0;
    switch (text[1])
    {
    case 's':
    case 'S':
        suit = 0;
        break;
    case 'h':
    case 'H':
        suit = 1;
        break;
    case 'd':
    case 'D':
        suit = 2;
        break;
    case 'c':
    case 'C':
        suit = 3;
        break;
    default:
        throw std::runtime_error("Invalid suit character");
    }
    return Card(suit + rank * 4);
}

HoleCards ParseHoleCards(const std::string& text)
{
    const std::vector<Card> cards = ParseCards(text);
    if (cards.size() != 2)
        throw std::runtime_error("Private hand must contain exactly 2 cards");
    return HoleCards(cards[0], cards[1]);
}

Board ParseBoard(const std::string& text, int expectedCardCount)
{
    const std::vector<Card> cards = ParseCards(text);
    if (expectedCardCount < 1 || expectedCardCount > 5)
        throw std::runtime_error("Expected public card count must be between 1 and 5");
    if (cards.size() != static_cast<std::size_t>(expectedCardCount))
        throw std::runtime_error("Public board contains an unexpected number of cards");
    return Board(cards);
}

std::string FormatCard(Card card)
{
    const int rank = card.Index() / 4;
    const int suit = card.Index() % 4;
    const std::string ranks = "AKQJT98765432";
    const std::string suits = "shdc";
    return {ranks[rank], suits[suit]};
}

bool Contains(HoleCards hand, Card card)
{
    return hand.CardAt(0) == card || hand.CardAt(1) == card;
}

bool Contains(const Board& board, Card card)
{
    for (int position = 0; position < board.CardCount(); ++position)
    {
        if (board.CardAt(position) == card)
            return true;
    }
    return false;
}

bool Overlaps(HoleCards first, HoleCards second)
{
    return Contains(first, second.CardAt(0)) || Contains(first, second.CardAt(1));
}

bool Overlaps(HoleCards hand, const Board& board)
{
    for (int position = 0; position < board.CardCount(); ++position)
    {
        if (Contains(hand, board.CardAt(position)))
            return true;
    }
    return false;
}
} // namespace solver::core
