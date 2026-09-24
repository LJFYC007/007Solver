#include "engine/HandTraversalData.h"
#include "engine/AverageStrategy.h"
#include "engine/ChanceGroups.h"
#include "game/TerminalSettlement.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
HandTraversalData::HandTraversalData(const SolveProblem& problem, game::NodeId root, bool prepareTraining)
    : HandTraversalData(std::make_shared<const HandBoardData>(problem, root), root, prepareTraining)
{}

HandTraversalData::HandTraversalData(std::shared_ptr<const HandBoardData> tables, game::NodeId root, bool prepareTraining)
    : tables(std::move(tables))
{
    const auto rootNode = this->tables->game->GetNode(root);
    const auto& rootState = rootNode.State();
    if (rootState.board != this->tables->board)
        throw std::invalid_argument("Hand tables belong to a different public board");
    const auto& hands = this->tables->hands;
    rootHalfPot = core::ToChipUnits(rootState.pot) / 2.0f;
    nodes.reserve(rootNode.TraversalNodeCount());
    children.reserve(rootNode.TraversalNodeCount() - 1);
    dealtCards.reserve(rootNode.TraversalNodeCount() - 1);
    const auto visit = [&](const auto& self, const game::GameNode& source, std::size_t depth) -> std::uint32_t
    {
        const auto& state = source.State();
        std::uint64_t boardMask = 0;
        for (int card = 0; card < state.board.CardCount(); ++card)
            boardMask |= std::uint64_t{1} << state.board.CardAt(card).Index();
        const auto kind = source.Kind() == game::NodeKind::Decision            ? Kind::Decision
                          : source.IsForcedRunout()                            ? Kind::ForcedRunout
                          : source.Kind() == game::NodeKind::Chance            ? Kind::Chance
                          : source.Terminal().kind == game::TerminalKind::Fold ? Kind::Fold
                                                                               : Kind::Showdown;
        Node node{source.Id(), kind, state.playerToAct.Index(), children.size(), 0, strategySize, boardMask, state.board};
        if (kind == Kind::Decision)
        {
            node.childCount = source.BettingEdgeCount();
            strategySize += node.childCount * hands[node.actor].size();
            for (const auto& hand : hands[node.actor])
                infoSetCount += !(hand.mask & boardMask);
            maxActions = std::max(maxActions, node.childCount);
        }
        else if (kind == Kind::Chance)
            node.childCount = source.ChanceOutcomeCount();
        else
        {
            game::TerminalSettlement settlement{
                state.pot, {rootState.stacks[0] - state.stacks[0], rootState.stacks[1] - state.stacks[1]}, std::nullopt
            };
            if (kind == Kind::Fold)
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
                if (kind == Kind::Showdown)
                    node.rankRow = this->tables->RankRow(state.board);
                else
                    runoutRows = std::max(runoutRows, RunoutRow(node) + 1);
            }
        }
        const auto index = static_cast<std::uint32_t>(nodes.size());
        nodes.push_back(node);
        children.resize(children.size() + node.childCount);
        dealtCards.resize(dealtCards.size() + node.childCount);
        maxDepth = std::max(maxDepth, depth + 1);
        for (std::size_t action = 0; action < node.childCount; ++action)
        {
            const auto child = source.Child(action);
            const auto childIndex = self(self, child, depth + 1);
            children[node.childOffset + action] = childIndex;
            if (node.kind == Kind::Chance)
                dealtCards[node.childOffset + action] =
                    static_cast<std::uint8_t>(child.State().board.CardAt(state.board.CardCount()).Index());
        }
        return index;
    };
    visit(visit, rootNode, 0);
    if (prepareTraining)
    {
        PrepareChanceTasks();
        PrepareRunoutOutcomes();
    }
}

void HandTraversalData::PrepareChanceTasks()
{
    VisitChanceGroups(
        tables->game->GetNode(nodes.front().id),
        [&](const game::GameNode& node, std::uint32_t index, const std::vector<std::uint32_t>& path)
        {
            const auto group = static_cast<std::uint32_t>(chanceGroups_.size());
            chanceGroups_.push_back({index, path});
            for (std::uint32_t action = 0; action < node.ChanceOutcomeCount(); ++action)
                chanceTasks_.push_back({group, action});
        }
    );
}

void HandTraversalData::PrepareRunoutOutcomes()
{
    if (!runoutRows)
        return;
    const auto& hands = tables->hands;
    const auto& masks = tables->handMasks;
    const std::size_t pairs = hands[0].size() * hands[1].size();
    runoutOutcomes_.assign(2 * runoutRows * pairs, 0);
    // Each rank row is one unordered pair of undealt cards: a runout of the flop row and,
    // with either card as the turn, a river of that card's turn row.
    for (int high = 0; high < 52; ++high)
        for (int low = 0; low < high; ++low)
        {
            const int row = tables->rowsByRunout[core::CardPairIndex(core::Card(high), core::Card(low))];
            if (row < 0)
                continue;
            std::uint32_t* targets[3] = {runoutOutcomes_.data()};
            std::size_t targetCount = 1;
            for (const int turn : {high, low})
                if (static_cast<std::size_t>(turn) + 1 < runoutRows)
                    targets[targetCount++] = runoutOutcomes_.data() + (static_cast<std::size_t>(turn) + 1) * pairs;
            const auto& ranks = tables->rankRows[row];
            for (std::size_t first = 0; first < ranks[0].hands.size(); ++first)
                for (std::size_t second = 0; second < ranks[1].hands.size(); ++second)
                {
                    const auto hand0 = ranks[0].hands[first], hand1 = ranks[1].hands[second];
                    if (masks[0][hand0] & masks[1][hand1])
                        continue;
                    const std::uint32_t outcome = (ranks[0].ranks[first] > ranks[1].ranks[second] ? 1u : 0u) |
                                                  (ranks[0].ranks[first] < ranks[1].ranks[second] ? 1u << 16 : 0u);
                    for (std::size_t target = 0; target < targetCount; ++target)
                        targets[target][hand0 * hands[1].size() + hand1] += outcome;
                }
        }
    std::uint32_t* transposed = runoutOutcomes_.data() + runoutRows * pairs;
    for (std::size_t row = 0; row < runoutRows; ++row)
        for (std::size_t hand0 = 0; hand0 < hands[0].size(); ++hand0)
            for (std::size_t hand1 = 0; hand1 < hands[1].size(); ++hand1)
                transposed[row * pairs + hand1 * hands[0].size() + hand0] = runoutOutcomes_[row * pairs + hand0 * hands[1].size() + hand1];
}

StrategySnapshot HandTraversalData::ExportStrategy(std::vector<float> sums) const
{
    std::vector<StrategySnapshot::NodeBlock> snapshotNodes;
    std::vector<core::HoleCards> snapshotHands;
    snapshotHands.reserve(infoSetCount);
    const auto maxHands = std::max(tables->hands[0].size(), tables->hands[1].size());
    std::vector<float> nodeSums(maxActions * maxHands);
    std::size_t writeOffset = 0;
    // Copy each action-major node before writing its compact hand-major output.
    // Output never extends beyond the original node block, so later inputs survive.
    for (const Node& node : nodes)
    {
        if (node.kind != Kind::Decision)
            continue;
        const auto& hands = tables->hands[node.actor];
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
    return StrategySnapshot(tables->game, std::move(snapshotNodes), std::move(snapshotHands), std::move(sums));
}
} // namespace solver::engine
