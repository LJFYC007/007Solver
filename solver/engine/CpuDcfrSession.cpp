#include "engine/CpuDcfrSession.h"
#include "game/TerminalSettlement.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <omp.h>

namespace solver::engine
{
CpuDcfrSession::CpuDcfrSession(std::shared_ptr<const SolveProblem> problem, int workers)
    : problem_(std::move(problem)), workerCount_(workers == 0 ? omp_get_max_threads() : workers)
{
    if (!problem_ || !problem_->game)
        throw std::invalid_argument("CPU DCFR session requires a solve problem");
    if (workerCount_ <= 0)
        throw std::invalid_argument("CPU DCFR worker count must be positive");

    const game::CompiledGame& game = *problem_->game;
    for (std::uint8_t player = 0; player < 2; ++player)
    {
        double totalWeight = 0.0;
        for (const auto& [cards, weight] : problem_->ranges.For(core::PlayerId(player)).Entries())
        {
            if (weight <= 0.0f || core::Overlaps(cards, game.Spec().initialBoard))
                continue;
            const auto pair = cards.Cards();
            const auto first = static_cast<std::uint8_t>(pair[0].Index());
            const auto second = static_cast<std::uint8_t>(pair[1].Index());
            hands_[player].push_back({cards, weight, (std::uint64_t{1} << first) | (std::uint64_t{1} << second), {first, second}});
            totalWeight += weight;
        }
        // A constant scale per player's range leaves regret matching unchanged.
        for (Hand& hand : hands_[player])
            hand.weight /= totalWeight;
    }

    bool hasLegalPair = false;
    for (std::size_t first = 0; first < hands_[0].size(); ++first)
    {
        for (std::size_t second = 0; second < hands_[1].size(); ++second)
        {
            if (!(hands_[0][first].mask & hands_[1][second].mask))
            {
                hasLegalPair = true;
                hands_[0][first].opponentMass += hands_[1][second].weight;
                hands_[1][second].opponentMass += hands_[0][first].weight;
            }
            if (hands_[0][first].cards == hands_[1][second].cards)
            {
                hands_[0][first].matchingOpponent = static_cast<int>(second);
                hands_[1][second].matchingOpponent = static_cast<int>(first);
            }
        }
    }
    if (!hasLegalPair)
        throw std::runtime_error("No valid private hand pairs after applying range weights and blockers");

    nodes_.reserve(game.NodeCount());
    std::vector<std::size_t> depths(game.NodeCount(), 0);
    std::array<int, 1326> rowsByRunout;
    rowsByRunout.fill(-1);
    std::size_t strategySize = 0;
    const float initialHalfPot = static_cast<float>(game.Spec().initialPot.Raw()) / (2.0f * core::Chips::kUnitsPerChip);
    for (std::size_t index = 0; index < game.NodeCount(); ++index)
    {
        const game::NodeId nodeId(static_cast<std::int32_t>(index));
        const auto& source = game.GetNode(nodeId);
        const auto& state = source.State();
        std::uint64_t boardMask = 0;
        for (int card = 0; card < state.board.CardCount(); ++card)
            boardMask |= std::uint64_t{1} << state.board.CardAt(card).Index();
        Node node{source.Kind(), state.playerToAct.Index(), children_.size(), 0, strategySize, boardMask};
        if (source.Parent())
            depths[index] = depths[source.Parent()->node.Value()] + 1;

        if (source.Kind() == game::NodeKind::Decision)
        {
            node.childCount = source.BettingEdgeCount();
            strategySize += node.childCount * hands_[node.actor].size();
            for (std::size_t edge = 0; edge < node.childCount; ++edge)
            {
                children_.push_back(static_cast<std::uint32_t>(source.GetBettingEdge(edge).NextNode().Value()));
                dealtCardMasks_.push_back(0);
            }
        }
        else if (source.Kind() == game::NodeKind::Chance)
        {
            node.childCount = source.ChanceOutcomeCount();
            for (std::size_t edge = 0; edge < node.childCount; ++edge)
            {
                const auto& outcome = source.GetChanceOutcome(edge);
                children_.push_back(static_cast<std::uint32_t>(outcome.NextNode().Value()));
                dealtCardMasks_.push_back(std::uint64_t{1} << outcome.DealtCard().Index());
            }
        }
        else
        {
            terminals_.push_back(static_cast<std::uint32_t>(index));
            game::TerminalSettlement settlement{
                state.pot,
                {game.Spec().initialStacks[0] - state.stacks[0], game.Spec().initialStacks[1] - state.stacks[1]},
                std::nullopt,
            };
            if (source.Terminal().kind == game::TerminalKind::Fold)
            {
                settlement.winner = source.Terminal().foldedPlayer->Other();
                node.utilities[0] = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - initialHalfPot;
            }
            else
            {
                for (std::size_t outcome = 0; outcome < 3; ++outcome)
                {
                    settlement.winner = outcome == 1 ? std::nullopt : std::optional<core::PlayerId>(core::PlayerId(outcome == 0 ? 0 : 1));
                    node.utilities[outcome] = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - initialHalfPot;
                }
                const int first = state.board.CardAt(3).Index();
                const int second = state.board.CardAt(4).Index();
                const int high = std::max(first, second);
                const int runout = high * (high - 1) / 2 + std::min(first, second);
                int& row = rowsByRunout[runout];
                if (row < 0)
                {
                    row = static_cast<int>(rankRows_.size());
                    rankRows_.emplace_back();
                    for (std::size_t player = 0; player < 2; ++player)
                    {
                        auto& ranks = rankRows_.back()[player];
                        for (std::size_t hand = 0; hand < hands_[player].size(); ++hand)
                        {
                            if (!(hands_[player][hand].mask & boardMask))
                                ranks.push_back({
                                    static_cast<std::uint16_t>(game.ShowdownRank(nodeId, hands_[player][hand].cards)),
                                    static_cast<std::uint16_t>(hand),
                                });
                        }
                        std::sort(
                            ranks.begin(),
                            ranks.end(),
                            [](const RankedHand& left, const RankedHand& right)
                            { return left.rank != right.rank ? left.rank < right.rank : left.hand < right.hand; }
                        );
                    }
                }
                node.rankRow = row;
            }
        }
        if (source.Kind() != game::NodeKind::Terminal)
        {
            if (levels_.size() <= depths[index])
                levels_.resize(depths[index] + 1);
            levels_[depths[index]].push_back(static_cast<std::uint32_t>(index));
        }
        nodes_.push_back(node);
    }

    regrets_.resize(strategySize, 0.0f);
    strategySums_.resize(strategySize, 0.0f);
    strategies_.resize(strategySize);
    for (const Node& node : nodes_)
    {
        if (node.kind == game::NodeKind::Decision)
        {
            std::fill_n(strategies_.data() + node.strategyOffset, node.childCount * hands_[node.actor].size(), 1.0f / node.childCount);
            for (const Hand& hand : hands_[node.actor])
                if (!(hand.mask & node.boardMask))
                    ++infoSetCount_;
        }
    }
    for (std::size_t player = 0; player < 2; ++player)
        reach_[player].resize(nodes_.size() * hands_[player].size());
    values_.resize(nodes_.size() * std::max(hands_[0].size(), hands_[1].size()));
}

void CpuDcfrSession::Run(int iterations, const std::function<void(int)>& progressCallback)
{
    if (iterations <= 0 || iterations > std::numeric_limits<int>::max() - completedIterations_)
        throw std::invalid_argument("DCFR iterations must be positive and fit the completed iteration counter");
    const auto start = std::chrono::steady_clock::now();
    auto lastProgress = start;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const std::size_t updatingPlayer = static_cast<std::size_t>(completedIterations_ % 2);
        const double t = completedIterations_ / 2 + 1.0;
        const double power = t * std::sqrt(t);
        const float positiveDiscount = static_cast<float>(power / (power + 1.0));
        const float averageDiscount = static_cast<float>((t / (t + 1.0)) * (t / (t + 1.0)));
        for (std::size_t player = 0; player < 2; ++player)
        {
            for (std::size_t hand = 0; hand < hands_[player].size(); ++hand)
                reach_[player][hand] = player == updatingPlayer ? 1.0f : hands_[player][hand].weight;
        }

#pragma omp parallel num_threads(workerCount_)
        {
#pragma omp master
            workerCount_ = omp_get_num_threads();
            for (std::size_t depth = 0; depth < levels_.size(); ++depth)
            {
                const auto& level = levels_[depth];
#pragma omp for schedule(static)
                for (int index = 0; index < static_cast<int>(level.size()); ++index)
                    PropagateReach(level[index], updatingPlayer);
            }
#pragma omp for schedule(static)
            for (int index = 0; index < static_cast<int>(terminals_.size()); ++index)
                EvaluateTerminal(terminals_[index], updatingPlayer);
            for (std::size_t depth = levels_.size(); depth > 0; --depth)
            {
                const auto& level = levels_[depth - 1];
#pragma omp for schedule(static)
                for (int index = 0; index < static_cast<int>(level.size()); ++index)
                    BackUp(level[index], updatingPlayer, positiveDiscount, averageDiscount);
            }
        }
        ++completedIterations_;
        if (progressCallback)
        {
            const auto now = std::chrono::steady_clock::now();
            if (iteration + 1 == iterations || now - lastProgress >= std::chrono::milliseconds(250))
            {
                progressCallback(completedIterations_);
                lastProgress = now;
            }
        }
    }
    trainingTimeSeconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void CpuDcfrSession::PropagateReach(std::uint32_t nodeIndex, std::size_t updatingPlayer)
{
    const Node& node = nodes_[nodeIndex];
    for (std::size_t player = 0; player < 2; ++player)
    {
        const std::size_t count = hands_[player].size();
        const double* parent = reach_[player].data() + nodeIndex * count;
        for (std::size_t action = 0; action < node.childCount; ++action)
        {
            const std::size_t edge = node.childOffset + action;
            double* child = reach_[player].data() + children_[edge] * count;
            if (node.kind == game::NodeKind::Chance)
            {
                // Public edges exclude board cards; remove the four private cards as well.
                const double chance = player == updatingPlayer ? 1.0 : 1.0 / static_cast<double>(node.childCount - 4);
                const auto mask = dealtCardMasks_[edge];
                for (std::size_t hand = 0; hand < count; ++hand)
                    child[hand] = hands_[player][hand].mask & mask ? 0.0f : parent[hand] * chance;
            }
            else if (node.actor == player)
            {
                const float* strategy = strategies_.data() + node.strategyOffset + action * count;
                for (std::size_t hand = 0; hand < count; ++hand)
                    child[hand] = parent[hand] * strategy[hand];
            }
            else
                std::copy_n(parent, count, child);
        }
    }
}

void CpuDcfrSession::EvaluateTerminal(std::uint32_t nodeIndex, std::size_t updatingPlayer)
{
    const Node& node = nodes_[nodeIndex];
    const auto& mine = hands_[updatingPlayer];
    const auto& opponent = hands_[1 - updatingPlayer];
    const double* opponentReach = reach_[1 - updatingPlayer].data() + nodeIndex * opponent.size();
    float* values = values_.data() + nodeIndex * mine.size();
    const float tie = updatingPlayer == 0 ? node.utilities[1] : -node.utilities[1];
    const float fold = updatingPlayer == 0 ? node.utilities[0] : -node.utilities[0];
    std::array<double, 52> cardMass{};
    double total = 0.0;
    for (std::size_t hand = 0; hand < opponent.size(); ++hand)
    {
        const double weight = opponentReach[hand];
        total += weight;
        cardMass[opponent[hand].cardIndices[0]] += weight;
        cardMass[opponent[hand].cardIndices[1]] += weight;
    }
    for (std::size_t index = 0; index < mine.size(); ++index)
    {
        const Hand& hand = mine[index];
        values[index] = 0.0f;
        if ((hand.mask & node.boardMask) || hand.opponentMass == 0.0)
            continue;
        const double identical = hand.matchingOpponent < 0 ? 0.0 : opponentReach[hand.matchingOpponent];
        double mass = total - cardMass[hand.cardIndices[0]] - cardMass[hand.cardIndices[1]] + identical;
        // Subtracting dominant blocked hands can erase tiny legal weights even in double.
        if (total > 0.0 && mass < 1e-6 * total)
        {
            mass = 0.0;
            for (std::size_t other = 0; other < opponent.size(); ++other)
                if (!(hand.mask & opponent[other].mask))
                    mass += opponentReach[other];
        }
        // This fixed per-hand scale preserves regret matching and avoids tiny root-pair masses.
        values[index] = static_cast<float>((mass / hand.opponentMass) * (node.rankRow < 0 ? fold : tie));
    }
    if (node.rankRow < 0)
        return;

    const auto& ranks = rankRows_[node.rankRow];
    const auto& myRanks = ranks[updatingPlayer];
    const auto& opponentRanks = ranks[1 - updatingPlayer];
    const float win = updatingPlayer == 0 ? node.utilities[0] : -node.utilities[2];
    const float loss = updatingPlayer == 0 ? node.utilities[2] : -node.utilities[0];
    // Strict rank comparisons exclude identical hands, so each blocked hand is subtracted once.
    total = 0.0;
    cardMass.fill(0.0);
    std::size_t cursor = 0;
    for (const RankedHand ranked : myRanks)
    {
        while (cursor < opponentRanks.size() && opponentRanks[cursor].rank < ranked.rank)
        {
            const auto index = opponentRanks[cursor++].hand;
            const double weight = opponentReach[index];
            total += weight;
            cardMass[opponent[index].cardIndices[0]] += weight;
            cardMass[opponent[index].cardIndices[1]] += weight;
        }
        const Hand& hand = mine[ranked.hand];
        double mass = total - cardMass[hand.cardIndices[0]] - cardMass[hand.cardIndices[1]];
        if (total > 0.0 && mass < 1e-6 * total)
        {
            mass = 0.0;
            for (std::size_t other = 0; other < cursor; ++other)
            {
                const auto index = opponentRanks[other].hand;
                if (!(hand.mask & opponent[index].mask))
                    mass += opponentReach[index];
            }
        }
        if (hand.opponentMass > 0.0)
            values[ranked.hand] += static_cast<float>((mass / hand.opponentMass) * (win - tie));
    }
    total = 0.0;
    cardMass.fill(0.0);
    cursor = opponentRanks.size();
    for (auto ranked = myRanks.rbegin(); ranked != myRanks.rend(); ++ranked)
    {
        while (cursor > 0 && opponentRanks[cursor - 1].rank > ranked->rank)
        {
            const auto index = opponentRanks[--cursor].hand;
            const double weight = opponentReach[index];
            total += weight;
            cardMass[opponent[index].cardIndices[0]] += weight;
            cardMass[opponent[index].cardIndices[1]] += weight;
        }
        const Hand& hand = mine[ranked->hand];
        double mass = total - cardMass[hand.cardIndices[0]] - cardMass[hand.cardIndices[1]];
        if (total > 0.0 && mass < 1e-6 * total)
        {
            mass = 0.0;
            for (std::size_t other = cursor; other < opponentRanks.size(); ++other)
            {
                const auto index = opponentRanks[other].hand;
                if (!(hand.mask & opponent[index].mask))
                    mass += opponentReach[index];
            }
        }
        if (hand.opponentMass > 0.0)
            values[ranked->hand] += static_cast<float>((mass / hand.opponentMass) * (loss - tie));
    }
}

void CpuDcfrSession::BackUp(std::uint32_t nodeIndex, std::size_t updatingPlayer, float positiveDiscount, float averageDiscount)
{
    const Node& node = nodes_[nodeIndex];
    const std::size_t count = hands_[updatingPlayer].size();
    float* values = values_.data() + nodeIndex * count;
    const bool updating = node.kind == game::NodeKind::Decision && node.actor == updatingPlayer;
    for (std::size_t hand = 0; hand < count; ++hand)
    {
        double value = 0.0;
        for (std::size_t action = 0; action < node.childCount; ++action)
        {
            const float childValue = values_[children_[node.childOffset + action] * count + hand];
            const float probability = updating ? strategies_[node.strategyOffset + action * count + hand] : 1.0f;
            value += static_cast<double>(probability) * childValue;
        }
        values[hand] = static_cast<float>(value);
    }
    if (!updating)
        return;

    const double* ownReach = reach_[updatingPlayer].data() + nodeIndex * count;
    for (std::size_t hand = 0; hand < count; ++hand)
    {
        float positiveRegret = 0.0f;
        for (std::size_t action = 0; action < node.childCount; ++action)
        {
            const std::size_t offset = node.strategyOffset + action * count + hand;
            const float regret = regrets_[offset] + (values_[children_[node.childOffset + action] * count + hand] - values[hand]);
            regrets_[offset] = regret * (regret > 0.0f ? positiveDiscount : 0.5f);
            positiveRegret += std::max(0.0f, regrets_[offset]);
            strategySums_[offset] = static_cast<float>(averageDiscount * (strategySums_[offset] + ownReach[hand] * strategies_[offset]));
        }
        for (std::size_t action = 0; action < node.childCount; ++action)
        {
            const std::size_t offset = node.strategyOffset + action * count + hand;
            strategies_[offset] = positiveRegret > 0.0f ? std::max(0.0f, regrets_[offset]) / positiveRegret : 1.0f / node.childCount;
        }
    }
}

StrategySnapshot CpuDcfrSession::ExportStrategy() const
{
    std::vector<game::InfoSetKey> infoSets;
    std::vector<float> probabilities;
    infoSets.reserve(infoSetCount_);
    probabilities.reserve(strategySums_.size());
    for (std::size_t index = 0; index < nodes_.size(); ++index)
    {
        const Node& node = nodes_[index];
        if (node.kind != game::NodeKind::Decision)
            continue;
        const auto& hands = hands_[node.actor];
        for (std::size_t hand = 0; hand < hands.size(); ++hand)
        {
            if (hands[hand].mask & node.boardMask)
                continue;
            double total = 0.0;
            for (std::size_t action = 0; action < node.childCount; ++action)
                total += strategySums_[node.strategyOffset + action * hands.size() + hand];
            infoSets.push_back({game::NodeId(static_cast<std::int32_t>(index)), hands[hand].cards});
            for (std::size_t action = 0; action < node.childCount; ++action)
                probabilities.push_back(
                    total > 0.0 ? static_cast<float>(strategySums_[node.strategyOffset + action * hands.size() + hand] / total)
                                : 1.0f / node.childCount
                );
        }
    }
    return StrategySnapshot(problem_->game, infoSets, std::move(probabilities));
}
} // namespace solver::engine
