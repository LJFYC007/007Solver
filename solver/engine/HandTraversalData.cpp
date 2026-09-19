#include "engine/HandTraversalData.h"
#include "engine/AverageStrategy.h"
#include "game/TerminalSettlement.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
HandTraversalData::HandTraversalData(const SolveProblem& problem, game::NodeId root, bool prepareTraining)
{
    if (!problem.game)
        throw std::invalid_argument("Hand traversal requires a compiled game");
    this->game = problem.game;
    const game::CompiledGame& game = *problem.game;
    const auto rootNode = game.GetNode(root);
    const auto& rootState = rootNode.State();
    for (std::uint8_t player = 0; player < 2; ++player)
    {
        float totalWeight = 0.0f;
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
        handMasks[player].resize(hands[player].size());
        for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
        {
            hands[player][hand].weight /= totalWeight;
            handMasks[player][hand] = hands[player][hand].mask;
        }
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
    rootHalfPot = rootState.pot.Raw() / (2.0f * core::Chips::kUnitsPerChip);
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
            std::vector<std::pair<std::uint16_t, std::uint16_t>> ranked;
            ranked.reserve(hands[player].size());
            for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
                if (!core::Overlaps(hands[player][hand].cards, board))
                    ranked.emplace_back(
                        static_cast<std::uint16_t>(game.ShowdownRank(board, hands[player][hand].cards)), static_cast<std::uint16_t>(hand)
                    );
            std::sort(ranked.begin(), ranked.end());
            auto& order = rankRows.back()[player];
            order.ranks.reserve(ranked.size());
            order.hands.reserve(ranked.size());
            order.card0.reserve(ranked.size());
            order.card1.reserve(ranked.size());
            order.masks.reserve(ranked.size());
            for (const auto& [rank, hand] : ranked)
            {
                const Hand& source = hands[player][hand];
                order.ranks.push_back(rank);
                order.hands.push_back(hand);
                order.card0.push_back(source.cardIndices[0]);
                order.card1.push_back(source.cardIndices[1]);
                order.masks.push_back(source.mask);
            }
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
            for (const auto& hand : hands[node.actor])
                infoSetCount += !(hand.mask & boardMask);
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
    if (prepareTraining)
    {
        PrepareChanceTasks();
        if (std::any_of(nodes.begin(), nodes.end(), [](const Node& node) { return node.forcedRunout && node.board.CardCount() == 3; }))
            PrepareFlopRunout();
    }
}

void HandTraversalData::PrepareChanceTasks()
{
    std::vector<std::uint32_t> path;
    const auto visit = [&](const auto& self, std::uint32_t index) -> void
    {
        const Node& node = nodes[index];
        if (node.forcedRunout || node.kind == game::NodeKind::Terminal)
            return;
        if (node.kind == game::NodeKind::Chance)
        {
            const auto group = static_cast<std::uint32_t>(chanceGroups_.size());
            chanceGroups_.push_back({index, path});
            for (std::uint32_t action = 0; action < node.childCount; ++action)
                chanceTasks_.push_back({group, action});
            return;
        }
        for (std::uint32_t action = 0; action < node.childCount; ++action)
        {
            path.push_back(action);
            self(self, children[node.childOffset + action]);
            path.pop_back();
        }
    };
    visit(visit, 0);
}

void HandTraversalData::PrepareFlopRunout()
{
    flopOutcomes_.resize(hands[0].size() * hands[1].size());
    for (const auto& ranks : rankRows)
        for (std::size_t first = 0; first < ranks[0].hands.size(); ++first)
            for (std::size_t second = 0; second < ranks[1].hands.size(); ++second)
            {
                if (ranks[0].masks[first] & ranks[1].masks[second])
                    continue;
                auto& outcomes = flopOutcomes_[ranks[0].hands[first] * hands[1].size() + ranks[1].hands[second]];
                outcomes.wins += ranks[0].ranks[first] > ranks[1].ranks[second];
                outcomes.losses += ranks[0].ranks[first] < ranks[1].ranks[second];
            }
}

StrategySnapshot HandTraversalData::ExportStrategy(std::vector<float> sums) const
{
    std::vector<StrategySnapshot::NodeBlock> snapshotNodes;
    std::vector<core::HoleCards> snapshotHands;
    snapshotHands.reserve(infoSetCount);
    const auto maxHands = std::max(hands[0].size(), hands[1].size());
    std::vector<float> nodeSums(maxActions * maxHands);
    std::size_t writeOffset = 0;
    // Copy each action-major node before writing its compact hand-major output.
    // Output never extends beyond the original node block, so later inputs survive.
    for (const Node& node : nodes)
    {
        if (node.kind != game::NodeKind::Decision)
            continue;
        const auto& hands = this->hands[node.actor];
        std::copy_n(sums.data() + node.strategyOffset, node.childCount * hands.size(), nodeSums.data());
        const auto handOffset = snapshotHands.size();
        const auto probabilityOffset = writeOffset;
        for (std::size_t hand = 0; hand < hands.size(); ++hand)
        {
            if (hands[hand].mask & node.boardMask)
                continue;
            snapshotHands.push_back(hands[hand].cards);
            NormalizeAverageStrategy(nodeSums.data() + hand, hands.size(), node.childCount, sums.data() + writeOffset, 1);
            writeOffset += node.childCount;
        }
        if (snapshotHands.size() != handOffset)
            snapshotNodes.push_back({node.id, handOffset, probabilityOffset, snapshotHands.size() - handOffset, node.childCount});
    }
    // Keep the allocation: shrinking capacity would require another large buffer.
    sums.resize(writeOffset);
    return StrategySnapshot(game, std::move(snapshotNodes), std::move(snapshotHands), std::move(sums));
}
} // namespace solver::engine
