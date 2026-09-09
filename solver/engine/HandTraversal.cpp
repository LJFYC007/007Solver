#include "engine/HandTraversal.h"
#include "game/TerminalSettlement.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace solver::engine
{
HandTraversal::HandTraversal(const SolveProblem& problem, game::NodeId root)
{
    if (!problem.game)
        throw std::invalid_argument("Hand traversal requires a compiled game");
    const game::CompiledGame& game = *problem.game;
    const auto& rootState = game.GetNode(root).State();
    for (std::uint8_t player = 0; player < 2; ++player)
    {
        double totalWeight = 0.0;
        for (const auto& [cards, weight] : problem.ranges.For(core::PlayerId(player)).Entries())
        {
            if (weight <= 0.0f || core::Overlaps(cards, rootState.board))
                continue;
            const auto pair = cards.Cards();
            const auto first = static_cast<std::uint8_t>(pair[0].Index());
            const auto second = static_cast<std::uint8_t>(pair[1].Index());
            hands[player].push_back({cards, weight, (std::uint64_t{1} << first) | (std::uint64_t{1} << second), {first, second}});
            totalWeight += weight;
        }
        // A constant scale per player's range leaves regret matching unchanged.
        for (Hand& hand : hands[player])
            hand.weight /= totalWeight;
    }

    bool hasLegalPair = false;
    for (std::size_t first = 0; first < hands[0].size(); ++first)
    {
        for (std::size_t second = 0; second < hands[1].size(); ++second)
        {
            if (!(hands[0][first].mask & hands[1][second].mask))
            {
                hasLegalPair = true;
                hands[0][first].opponentMass += hands[1][second].weight;
                hands[1][second].opponentMass += hands[0][first].weight;
            }
            if (hands[0][first].cards == hands[1][second].cards)
            {
                hands[0][first].matchingOpponent = static_cast<int>(second);
                hands[1][second].matchingOpponent = static_cast<int>(first);
            }
        }
    }
    if (!hasLegalPair)
        throw std::runtime_error("No valid private hand pairs after applying range weights and blockers");

    std::vector<game::NodeId> sourceIds{root};
    std::vector<std::size_t> depths{0};
    std::array<int, 1326> rowsByRunout;
    rowsByRunout.fill(-1);
    rootHalfPot = static_cast<float>(rootState.pot.Raw()) / (2.0f * core::Chips::kUnitsPerChip);
    for (std::size_t index = 0; index < sourceIds.size(); ++index)
    {
        const game::NodeId nodeId = sourceIds[index];
        const auto& source = game.GetNode(nodeId);
        const auto& state = source.State();
        std::uint64_t boardMask = 0;
        for (int card = 0; card < state.board.CardCount(); ++card)
            boardMask |= std::uint64_t{1} << state.board.CardAt(card).Index();
        Node node{nodeId, source.Kind(), state.playerToAct.Index(), children.size(), 0, strategySize, boardMask};

        if (source.Kind() == game::NodeKind::Decision)
        {
            node.childCount = source.BettingEdgeCount();
            strategySize += node.childCount * hands[node.actor].size();
            for (std::size_t edge = 0; edge < node.childCount; ++edge)
            {
                children.push_back(static_cast<std::uint32_t>(sourceIds.size()));
                sourceIds.push_back(source.GetBettingEdge(edge).NextNode());
                depths.push_back(depths[index] + 1);
                dealtCardMasks.push_back(0);
            }
        }
        else if (source.Kind() == game::NodeKind::Chance)
        {
            node.childCount = source.ChanceOutcomeCount();
            for (std::size_t edge = 0; edge < node.childCount; ++edge)
            {
                const auto& outcome = source.GetChanceOutcome(edge);
                children.push_back(static_cast<std::uint32_t>(sourceIds.size()));
                sourceIds.push_back(outcome.NextNode());
                depths.push_back(depths[index] + 1);
                dealtCardMasks.push_back(std::uint64_t{1} << outcome.DealtCard().Index());
            }
        }
        else
        {
            terminals.push_back(static_cast<std::uint32_t>(index));
            game::TerminalSettlement settlement{
                state.pot,
                {rootState.stacks[0] - state.stacks[0], rootState.stacks[1] - state.stacks[1]},
                std::nullopt,
            };
            if (source.Terminal().kind == game::TerminalKind::Fold)
            {
                settlement.winner = source.Terminal().foldedPlayer->Other();
                node.utilities[0] = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - rootHalfPot;
            }
            else
            {
                for (std::size_t outcome = 0; outcome < 3; ++outcome)
                {
                    settlement.winner = outcome == 1 ? std::nullopt : std::optional<core::PlayerId>(core::PlayerId(outcome == 0 ? 0 : 1));
                    node.utilities[outcome] = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - rootHalfPot;
                }
                const int first = state.board.CardAt(3).Index();
                const int second = state.board.CardAt(4).Index();
                const int high = std::max(first, second);
                const int runout = high * (high - 1) / 2 + std::min(first, second);
                int& row = rowsByRunout[runout];
                if (row < 0)
                {
                    row = static_cast<int>(rankRows.size());
                    rankRows.emplace_back();
                    for (std::size_t player = 0; player < 2; ++player)
                    {
                        auto& ranks = rankRows.back()[player];
                        for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
                        {
                            if (!(hands[player][hand].mask & boardMask))
                                ranks.push_back({
                                    static_cast<std::uint16_t>(game.ShowdownRank(nodeId, hands[player][hand].cards)),
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
            if (levels.size() <= depths[index])
                levels.resize(depths[index] + 1);
            levels[depths[index]].push_back(static_cast<std::uint32_t>(index));
        }
        nodes.push_back(node);
    }
}

std::vector<float> HandTraversal::LoadStrategy(const StrategySnapshot& strategy) const
{
    std::vector<float> packed(strategySize);
    for (const Node& node : nodes)
    {
        if (node.kind != game::NodeKind::Decision)
            continue;
        const auto& actorHands = hands[node.actor];
        const float uniform = 1.0f / static_cast<float>(node.childCount);
        for (std::size_t hand = 0; hand < actorHands.size(); ++hand)
        {
            const float* probabilities = strategy.FindStrategy({node.id, actorHands[hand].cards});
            for (std::size_t action = 0; action < node.childCount; ++action)
                packed[node.strategyOffset + action * actorHands.size() + hand] = probabilities ? probabilities[action] : uniform;
        }
    }
    return packed;
}

std::vector<double> HandTraversal::OpponentReachAtRoot(const StrategySnapshot& strategy, std::size_t opponentPlayer) const
{
    const auto& game = strategy.Game();
    std::vector<game::ParentEdge> path;
    for (auto node = nodes.front().id; game.GetNode(node).Parent();)
    {
        const auto parent = *game.GetNode(node).Parent();
        path.push_back(parent);
        node = parent.node;
    }
    std::vector<double> reach;
    for (const Hand& hand : hands[opponentPlayer])
    {
        double weight = hand.weight;
        for (auto step = path.rbegin(); step != path.rend(); ++step)
        {
            const auto& node = game.GetNode(step->node);
            if (node.Kind() == game::NodeKind::Decision && node.State().playerToAct.Index() == opponentPlayer)
            {
                const float* probabilities = strategy.FindStrategy({step->node, hand.cards});
                weight *= probabilities ? probabilities[step->edgeIndex] : 1.0f / static_cast<float>(node.BettingEdgeCount());
            }
        }
        // The root board already removes dealt cards. Earlier chance probabilities and
        // the queried hand's own reach cancel in its conditional opponent distribution.
        reach.push_back(weight);
    }
    return reach;
}

std::vector<double> HandTraversal::CompatibleMasses(std::size_t player, const double* opponentReach) const
{
    std::vector<double> masses(hands[player].size(), 0.0);
    for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
        for (std::size_t other = 0; other < hands[1 - player].size(); ++other)
            if (!(hands[player][hand].mask & hands[1 - player][other].mask))
                masses[hand] += opponentReach[other];
    return masses;
}

void HandTraversal::PropagateReach(
    std::uint32_t nodeIndex,
    std::size_t player,
    bool includeChance,
    const float* strategy,
    const double* parent,
    double* reaches,
    bool includeTerminals
) const
{
    const Node& node = nodes[nodeIndex];
    const std::size_t count = hands[player].size();
    for (std::size_t action = 0; action < node.childCount; ++action)
    {
        const std::size_t edge = node.childOffset + action;
        if (!includeTerminals && nodes[children[edge]].kind == game::NodeKind::Terminal)
            continue;
        double* child = reaches + children[edge] * count;
        if (node.kind == game::NodeKind::Chance)
        {
            // Public edges exclude board cards; remove the four private cards as well.
            const double chance = !includeChance ? 1.0 : 1.0 / static_cast<double>(node.childCount - 4);
            const auto mask = dealtCardMasks[edge];
            for (std::size_t hand = 0; hand < count; ++hand)
                child[hand] = hands[player][hand].mask & mask ? 0.0f : parent[hand] * chance;
        }
        else if (node.actor == player)
        {
            const float* probabilities = strategy + node.strategyOffset + action * count;
            for (std::size_t hand = 0; hand < count; ++hand)
                child[hand] = parent[hand] * probabilities[hand];
        }
        else
            std::copy_n(parent, count, child);
    }
}

void HandTraversal::EvaluateTerminal(
    std::uint32_t nodeIndex,
    std::size_t updatingPlayer,
    const double* opponentReach,
    const double* divisors,
    float* values
) const
{
    const Node& node = nodes[nodeIndex];
    const auto& mine = hands[updatingPlayer];
    const auto& opponent = hands[1 - updatingPlayer];
    const float tie = updatingPlayer == 0 ? node.utilities[1] : -node.utilities[1];
    const float fold = updatingPlayer == 0 ? node.utilities[0] : -node.utilities[0];
    std::array<double, 52> cardMass{};
    double total = 0.0;
    // Zero-tie showdowns need only the win/loss sweeps below. Other subtree roots
    // can have nonzero tie utility, so retain their compatible-mass baseline.
    if (node.rankRow >= 0 && tie == 0.0f)
        std::fill_n(values, mine.size(), 0.0f);
    else
    {
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
            if ((hand.mask & node.boardMask) || divisors[index] == 0.0)
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
            values[index] = static_cast<float>((mass / divisors[index]) * (node.rankRow < 0 ? fold : tie));
        }
    }
    if (node.rankRow < 0)
        return;

    const auto& ranks = rankRows[node.rankRow];
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
        if (divisors[ranked.hand] > 0.0)
            values[ranked.hand] += static_cast<float>((mass / divisors[ranked.hand]) * (win - tie));
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
        if (divisors[ranked->hand] > 0.0)
            values[ranked->hand] += static_cast<float>((mass / divisors[ranked->hand]) * (loss - tie));
    }
}

void HandTraversal::BackUp(std::uint32_t nodeIndex, std::size_t player, const float* strategy, bool bestResponse, float* values) const
{
    const Node& node = nodes[nodeIndex];
    const std::size_t count = hands[player].size();
    const bool acting = node.kind == game::NodeKind::Decision && node.actor == player;
    // Local scratch keeps independent hands contiguous without changing reduction precision or order.
    std::array<double, kMaxHands> accumulated;
    std::fill_n(accumulated.data(), count, acting && bestResponse ? -std::numeric_limits<double>::infinity() : 0.0);
    for (std::size_t action = 0; action < node.childCount; ++action)
    {
        const float* child = values + children[node.childOffset + action] * count;
        if (acting && bestResponse)
        {
            for (std::size_t hand = 0; hand < count; ++hand)
                accumulated[hand] = std::max(accumulated[hand], static_cast<double>(child[hand]));
        }
        else if (acting)
        {
            const float* probability = strategy + node.strategyOffset + action * count;
            for (std::size_t hand = 0; hand < count; ++hand)
                accumulated[hand] += static_cast<double>(probability[hand]) * child[hand];
        }
        else
        {
            for (std::size_t hand = 0; hand < count; ++hand)
                accumulated[hand] += static_cast<double>(child[hand]);
        }
    }
    for (std::size_t hand = 0; hand < count; ++hand)
        values[nodeIndex * count + hand] = static_cast<float>(accumulated[hand]);
}
} // namespace solver::engine
