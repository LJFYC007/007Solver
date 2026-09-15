#include "engine/HandTraversal.h"
#include "game/TerminalSettlement.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <omp.h>

namespace solver::engine
{
HandTraversal::HandTraversal(const SolveProblem& problem, game::NodeId root, bool cacheFlopRunout)
{
    if (!problem.game)
        throw std::invalid_argument("Hand traversal requires a compiled game");
    const game::CompiledGame& game = *problem.game;
    const auto rootNode = game.GetNode(root);
    const auto& rootState = rootNode.State();
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

    rowsByRunout.fill(-1);
    rootHalfPot = static_cast<float>(rootState.pot.Raw()) / (2.0f * core::Chips::kUnitsPerChip);
    // Rank rows are shared by all betting histories and reversed turn/river runouts.
    const auto addRanks = [&](const core::Board& board)
    {
        const int first = board.CardAt(3).Index();
        const int second = board.CardAt(4).Index();
        const int high = std::max(first, second);
        const int key = high * (high - 1) / 2 + std::min(first, second);
        if (rowsByRunout[key] >= 0)
            return;
        rowsByRunout[key] = static_cast<int>(rankRows.size());
        rankRows.emplace_back();
        for (std::size_t player = 0; player < 2; ++player)
        {
            auto& ranks = rankRows.back()[player];
            ranks.reserve(hands[player].size());
            for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
                if (!core::Overlaps(hands[player][hand].cards, board))
                    ranks.push_back(
                        {static_cast<std::uint16_t>(game.ShowdownRank(board, hands[player][hand].cards)), static_cast<std::uint16_t>(hand)}
                    );
            std::sort(
                ranks.begin(),
                ranks.end(),
                [](const RankedHand& a, const RankedHand& b) { return a.rank != b.rank ? a.rank < b.rank : a.hand < b.hand; }
            );
        }
    };
    if (rootState.board.CardCount() == 5)
        addRanks(rootState.board);
    else
        for (int first = 0; first < 52; ++first)
        {
            if (core::Contains(rootState.board, core::Card(first)))
                continue;
            const auto board = rootState.board.Append(core::Card(first));
            if (board.CardCount() == 5)
                addRanks(board);
            else
                for (int second = 0; second < first; ++second)
                    if (!core::Contains(board, core::Card(second)))
                        addRanks(board.Append(core::Card(second)));
        }
    nodes.reserve(rootNode.TraversalNodeCount());
    children.reserve(rootNode.TraversalNodeCount() - 1);
    dealtCardMasks.reserve(rootNode.TraversalNodeCount() - 1);
    const auto visit = [&](const auto& self, const game::GameNode& source, std::size_t depth) -> std::uint32_t
    {
        const auto& state = source.State();
        std::uint64_t boardMask = 0;
        for (int card = 0; card < state.board.CardCount(); ++card)
            boardMask |= std::uint64_t{1} << state.board.CardAt(card).Index();
        Node node{source.Id(), source.Kind(), state.playerToAct.Index(), children.size(), 0, strategySize, boardMask, state.board};
        node.forcedRunout = source.IsForcedRunout();
        if (node.kind == game::NodeKind::Decision)
        {
            node.childCount = source.BettingEdgeCount();
            strategySize += node.childCount * hands[node.actor].size();
            maxActions = std::max(maxActions, node.childCount);
        }
        else if (node.kind == game::NodeKind::Chance && !node.forcedRunout)
            node.childCount = source.ChanceOutcomeCount();
        else
        {
            game::TerminalSettlement settlement{
                state.pot, {rootState.stacks[0] - state.stacks[0], rootState.stacks[1] - state.stacks[1]}, std::nullopt
            };
            if (!node.forcedRunout && source.Terminal().kind == game::TerminalKind::Fold)
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
                if (!node.forcedRunout)
                {
                    const int a = state.board.CardAt(3).Index(), b = state.board.CardAt(4).Index();
                    node.rankRow = rowsByRunout[std::max(a, b) * (std::max(a, b) - 1) / 2 + std::min(a, b)];
                }
            }
        }
        const auto index = static_cast<std::uint32_t>(nodes.size());
        nodes.push_back(node);
        children.resize(children.size() + node.childCount);
        dealtCardMasks.resize(dealtCardMasks.size() + node.childCount);
        maxDepth = std::max(maxDepth, depth + 1);
        for (std::size_t action = 0; action < node.childCount; ++action)
        {
            const auto child = source.Child(action);
            const auto childIndex = self(self, child, depth + 1);
            children[node.childOffset + action] = childIndex;
            if (node.kind == game::NodeKind::Chance)
                dealtCardMasks[node.childOffset + action] = std::uint64_t{1} << child.State().board.CardAt(state.board.CardCount()).Index();
        }
        return index;
    };
    visit(visit, rootNode, 0);
    if (cacheFlopRunout &&
        std::any_of(nodes.begin(), nodes.end(), [](const Node& node) { return node.forcedRunout && node.board.CardCount() == 3; }))
        PrepareFlopRunout();
}

void HandTraversal::PrepareFlopRunout()
{
    flopOutcomes_.resize(hands[0].size() * hands[1].size());
    for (const auto& ranks : rankRows)
        for (const RankedHand first : ranks[0])
            for (const RankedHand second : ranks[1])
            {
                if (hands[0][first.hand].mask & hands[1][second.hand].mask)
                    continue;
                auto& outcomes = flopOutcomes_[first.hand * hands[1].size() + second.hand];
                outcomes.wins += first.rank > second.rank;
                outcomes.losses += first.rank < second.rank;
            }
}

void HandTraversal::EvaluateFlopRunout(
    const Node& node,
    std::size_t player,
    const double* opponentReach,
    const double* divisors,
    float* values
) const
{
    const float tie = player == 0 ? node.utilities[1] : -node.utilities[1];
    const float win = player == 0 ? node.utilities[0] : -node.utilities[2];
    const float loss = player == 0 ? node.utilities[2] : -node.utilities[0];
    const double winScale = static_cast<double>(win - tie) / 990.0;
    const double lossScale = static_cast<double>(loss - tie) / 990.0;
    std::array<double, kMaxHands> wins{}, losses{};
    // Both orientations read contiguous rows from the same immutable table.
    for (std::size_t first = 0; first < hands[0].size(); ++first)
        for (std::size_t second = 0; second < hands[1].size(); ++second)
        {
            const auto outcomes = flopOutcomes_[first * hands[1].size() + second];
            const auto hand = player == 0 ? first : second;
            const double reach = opponentReach[player == 0 ? second : first];
            wins[hand] += reach * (player == 0 ? outcomes.wins : outcomes.losses);
            losses[hand] += reach * (player == 0 ? outcomes.losses : outcomes.wins);
        }
    const auto masses = tie == 0.0f ? std::vector<double>{} : CompatibleMasses(player, opponentReach);
    for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
    {
        const double baseline = tie == 0.0f ? 0.0 : masses[hand] * tie;
        values[hand] = divisors[hand] > 0.0
                           ? static_cast<float>((baseline + wins[hand] * winScale + losses[hand] * lossScale) / divisors[hand])
                           : 0.0f;
    }
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

void HandTraversal::PropagateChild(
    std::uint32_t nodeIndex,
    std::size_t action,
    std::size_t player,
    bool includeChance,
    const float* strategy,
    const double* parent,
    double* child
) const
{
    const Node& node = nodes[nodeIndex];
    const std::size_t count = hands[player].size();
    if (node.kind == game::NodeKind::Chance)
    {
        const double chance = includeChance ? 1.0 / static_cast<double>(node.childCount - 4) : 1.0;
        const auto mask = dealtCardMasks[node.childOffset + action];
        for (std::size_t hand = 0; hand < count; ++hand)
            child[hand] = hands[player][hand].mask & mask ? 0.0 : parent[hand] * chance;
    }
    else if (node.actor == player)
    {
        const float* probabilities = strategy + action * count;
        for (std::size_t hand = 0; hand < count; ++hand)
            child[hand] = parent[hand] * probabilities[hand];
    }
    else
        std::copy_n(parent, count, child);
}

void HandTraversal::EvaluateTerminal(
    const Node& node,
    std::size_t updatingPlayer,
    const double* opponentReach,
    const double* divisors,
    float* values
) const
{
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

void HandTraversal::EvaluateRunout(Node node, std::size_t player, const double* opponentReach, const double* divisors, float* values) const
{
    if (node.board.CardCount() == 3 && !flopOutcomes_.empty())
    {
        EvaluateFlopRunout(node, player, opponentReach, divisors, values);
        return;
    }
    if (node.board.CardCount() == 5)
    {
        const int a = node.board.CardAt(3).Index(), b = node.board.CardAt(4).Index();
        node.rankRow = rowsByRunout[std::max(a, b) * (std::max(a, b) - 1) / 2 + std::min(a, b)];
        EvaluateTerminal(node, player, opponentReach, divisors, values);
        return;
    }
    std::array<double, kMaxHands> accumulated{};
    std::array<double, kMaxHands> childReach;
    std::array<float, kMaxHands> childValues;
    const double chance = 1.0 / (52 - node.board.CardCount() - 4);
    for (int card = 0; card < 52; ++card)
    {
        const auto mask = std::uint64_t{1} << card;
        if (node.boardMask & mask)
            continue;
        Node child = node;
        child.board = node.board.Append(core::Card(card));
        child.boardMask |= mask;
        for (std::size_t hand = 0; hand < hands[1 - player].size(); ++hand)
            childReach[hand] = hands[1 - player][hand].mask & mask ? 0.0 : opponentReach[hand] * chance;
        EvaluateRunout(child, player, childReach.data(), divisors, childValues.data());
        for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
            accumulated[hand] += childValues[hand];
    }
    for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
        values[hand] = static_cast<float>(accumulated[hand]);
}

HandTraversal::Workspace HandTraversal::MakeWorkspace(bool parallel) const
{
    Workspace workspace;
    const auto count = std::max(hands[0].size(), hands[1].size());
    for (std::size_t player = 0; player < 2; ++player)
        workspace.reach[player].resize((maxDepth + 1) * hands[player].size());
    workspace.childValues.resize(maxDepth * maxActions * count);
    workspace.accumulated.resize(maxDepth * count);
    if (parallel)
        workspace.parallelValues.resize(52 * count);
    workspace.strategies.resize(maxDepth * maxActions * count);
    return workspace;
}

void HandTraversal::Walk(
    std::uint32_t nodeIndex,
    std::size_t player,
    const StrategySnapshot* strategy,
    const double* divisors,
    bool bestResponse,
    Workspace& workspace,
    std::size_t depth,
    float* values,
    const Update& update,
    std::vector<Workspace>* workers,
    const float* regrets
) const
{
    const Node& node = nodes[nodeIndex];
    const auto count = hands[player].size();
    const auto stride = std::max(hands[0].size(), hands[1].size());
    const auto opponentCount = hands[1 - player].size();
    const double* opponentReach = workspace.reach[1 - player].data() + depth * opponentCount;
    if (node.forcedRunout)
    {
        EvaluateRunout(node, player, opponentReach, divisors, values);
        return;
    }
    if (node.kind == game::NodeKind::Terminal)
    {
        EvaluateTerminal(node, player, opponentReach, divisors, values);
        return;
    }
    const float* nodeStrategy = nullptr;
    if (node.kind == game::NodeKind::Decision)
    {
        const auto actorCount = hands[node.actor].size();
        float* current = workspace.strategies.data() + depth * maxActions * stride;
        if (regrets)
        {
            // Preserve this entry strategy until reach, backup and average-strategy
            // accumulation finish. Descendants and other workers use separate rows.
            std::array<float, kMaxHands> positiveRegrets;
            std::fill_n(positiveRegrets.data(), actorCount, 0.0f);
            for (std::size_t action = 0; action < node.childCount; ++action)
                for (std::size_t hand = 0; hand < actorCount; ++hand)
                {
                    const auto offset = action * actorCount + hand;
                    current[offset] = std::max(0.0f, regrets[node.strategyOffset + offset]);
                    positiveRegrets[hand] += current[offset];
                }
            for (std::size_t action = 0; action < node.childCount; ++action)
                for (std::size_t hand = 0; hand < actorCount; ++hand)
                {
                    const auto offset = action * actorCount + hand;
                    current[offset] = positiveRegrets[hand] > 0.0f ? current[offset] / positiveRegrets[hand] : 1.0f / node.childCount;
                }
        }
        else
        {
            const auto source = strategy->FindNodeStrategy(node.id);
            const float uniform = 1.0f / static_cast<float>(node.childCount);
            std::fill_n(current, node.childCount * actorCount, uniform);
            if (source)
            {
                // Both hand lists are sorted; merge once rather than looking up each infoset.
                std::size_t other = 0;
                for (std::size_t hand = 0; hand < actorCount; ++hand)
                {
                    const auto cards = hands[node.actor][hand].cards;
                    while (other < source->handCount && source->hands[other] < cards)
                        ++other;
                    if (other < source->handCount && source->hands[other] == cards)
                        for (std::size_t action = 0; action < node.childCount; ++action)
                            current[action * actorCount + hand] = source->probabilities[other * source->actionCount + action];
                }
            }
        }
        nodeStrategy = current;
    }
    const bool acting = node.kind == game::NodeKind::Decision && node.actor == player;
    const bool parallel = node.kind == game::NodeKind::Chance && workers && !workers->empty();
    float* childrenValues = parallel ? workspace.parallelValues.data() : workspace.childValues.data() + depth * maxActions * stride;
    double* accumulated = workspace.accumulated.data() + depth * stride;
    std::fill_n(accumulated, count, acting && bestResponse ? -std::numeric_limits<double>::infinity() : 0.0);
    const auto descend = [&](std::size_t action, Workspace& scratch, float* output, std::vector<Workspace>* pool)
    {
        for (std::size_t p = 0; p < 2; ++p)
            if (p != player || update)
                PropagateChild(
                    nodeIndex,
                    action,
                    p,
                    p != player,
                    nodeStrategy,
                    workspace.reach[p].data() + depth * hands[p].size(),
                    scratch.reach[p].data() + (depth + 1) * hands[p].size()
                );
        Walk(
            children[node.childOffset + action], player, strategy, divisors, bestResponse, scratch, depth + 1, output, update, pool, regrets
        );
    };
    if (parallel)
    {
#pragma omp parallel for num_threads(static_cast<int>(workers->size())) schedule(dynamic, 1)
        for (int action = 0; action < static_cast<int>(node.childCount); ++action)
            descend(static_cast<std::size_t>(action), (*workers)[omp_get_thread_num()], childrenValues + action * count, nullptr);
    }
    for (std::size_t action = 0; action < node.childCount; ++action)
    {
        // Serial chance branches reuse one row; decision rows survive until the regret update.
        float* child = childrenValues + ((parallel || node.kind == game::NodeKind::Decision) ? action * count : 0);
        if (!parallel)
            descend(action, workspace, child, workers);
        const float* probability = acting ? nodeStrategy + action * count : nullptr;
        for (std::size_t hand = 0; hand < count; ++hand)
            if (acting && bestResponse)
                accumulated[hand] = std::max(accumulated[hand], static_cast<double>(child[hand]));
            else
                accumulated[hand] += acting ? static_cast<double>(probability[hand]) * child[hand] : child[hand];
    }
    for (std::size_t hand = 0; hand < count; ++hand)
        values[hand] = static_cast<float>(accumulated[hand]);
    if (acting && update)
        update(nodeIndex, workspace.reach[player].data() + depth * count, nodeStrategy, childrenValues, values);
}
} // namespace solver::engine
