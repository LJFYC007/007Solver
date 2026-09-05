#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace solver::core
{
class Card
{
public:
    explicit Card(int index);

    int Index() const { return index_; }

    friend bool operator==(Card left, Card right) { return left.index_ == right.index_; }
    friend bool operator!=(Card left, Card right) { return !(left == right); }
    friend bool operator<(Card left, Card right) { return left.index_ < right.index_; }

private:
    int index_;
};

class HoleCards
{
public:
    HoleCards(Card first, Card second);

    Card CardAt(int position) const;
    std::array<Card, 2> Cards() const;

    friend bool operator==(HoleCards left, HoleCards right) { return left.packed_ == right.packed_; }
    friend bool operator!=(HoleCards left, HoleCards right) { return !(left == right); }
    friend bool operator<(HoleCards left, HoleCards right) { return left.packed_ < right.packed_; }

private:
    friend struct HoleCardsHash;

    int packed_;
};

struct HoleCardsHash
{
    std::size_t operator()(HoleCards cards) const noexcept { return std::hash<int>{}(cards.packed_); }
};

class Board
{
public:
    explicit Board(const std::vector<Card>& cards);

    int CardCount() const { return cardCount_; }
    Card CardAt(int dealOrderIndex) const;
    Board Append(Card card) const;

    friend bool operator==(const Board& left, const Board& right)
    {
        return left.packed_ == right.packed_ && left.cardCount_ == right.cardCount_;
    }
    friend bool operator!=(const Board& left, const Board& right) { return !(left == right); }
    friend bool operator<(const Board& left, const Board& right)
    {
        return left.cardCount_ < right.cardCount_ || (left.cardCount_ == right.cardCount_ && left.packed_ < right.packed_);
    }

private:
    Board(int packed, int cardCount);

    int packed_;
    int cardCount_;
};

Card ParseCard(const std::string& text);
HoleCards ParseHoleCards(const std::string& text);
Board ParseBoard(const std::string& text, int expectedCardCount);
std::string FormatCard(Card card);

bool Contains(HoleCards hand, Card card);
bool Contains(const Board& board, Card card);
bool Overlaps(HoleCards first, HoleCards second);
bool Overlaps(HoleCards hand, const Board& board);
} // namespace solver::core
