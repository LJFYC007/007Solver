#include "engine/HandTraversalData.h"
#include "engine/AverageStrategy.h"
#include "engine/ChanceGroups.h"
#include "engine/gpu/GpuQuantize.h"
#include "game/TerminalSettlement.h"
#include <algorithm>
#include <unordered_map>
#include <utility>

namespace solver::engine
{
HandTraversalData::HandTraversalData(const SolveProblem& problem, game::NodeId root, bool prepareTraining) : tables(problem, root)
{
    const auto rootNode = tables.game->GetNode(root);
    const auto& rootState = rootNode.State();
    const auto& hands = tables.hands;
    rootHalfPot = core::ToChipUnits(rootState.pot) / 2.0f;
    // The subtrees below each betting path's first non-all-in chance node (ChanceGroups.h) are
    // visited in parallel, each into the preorder ranges that the sequential visit of the nodes
    // above them reserves from the subtrees' node, decision and action counts.
    // The unpadded entries (Node::strategyOffset) and state units of an actor's decisions with
    // these actions in total.
    const auto decisionUnits = [&](std::size_t actions, std::size_t decisions, std::size_t actor)
    {
        const auto count = hands[actor].size();
        return std::array<std::size_t, 2>{actions * count + decisions * ((count + 1) / 2), StateUnits(actions, count, decisions)};
    };
    // Where a visit writes its next node, child edge, unpadded entry and state unit, and the
    // extremes it saw.
    struct Cursor
    {
        std::uint32_t node = 0;
        std::size_t child = 0, strategy = 0, state = 0, maxActions = 1, maxDepth = 1, runoutRows = 0;
    };
    // A subtree the visit of the nodes above defers: its cursor, depth and root.
    struct Start
    {
        Cursor at;
        std::size_t depth;
        game::GameNode root;
    };
    std::vector<Start> starts;
    // Placeholders, each overwritten by the visit of its node.
    nodes.assign(rootNode.TraversalNodeCount(), Node{rootNode.Id(), Kind::Fold, 0, 0, 0, 0, 0, rootState.board});
    children.resize(nodes.size() - 1);
    dealtCards.resize(nodes.size() - 1);
    const auto visit = [&](const auto& self, const game::GameNode& source, std::size_t depth, Cursor& at, bool above) -> std::uint32_t
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
        Node node{source.Id(), kind, state.playerToAct.Index(), at.child, 0, at.strategy, boardMask, state.board};
        if (kind == Kind::Decision)
        {
            node.childCount = source.BettingEdgeCount();
            node.stateOffset = at.state;
            const auto units = decisionUnits(node.childCount, 1, node.actor);
            at.strategy += units[0];
            at.state += units[1];
            at.maxActions = std::max(at.maxActions, node.childCount);
        }
        else if (kind == Kind::Chance)
            node.childCount = source.ChanceOutcomeCount();
        else
        {
            game::TerminalSettlement settlement{
                state.pot,
                {rootState.stacks[0] - state.stacks[0], rootState.stacks[1] - state.stacks[1]},
                std::nullopt,
                game::TerminalRake(tables.game->Spec(), state),
            };
            node.rake = settlement.rake;
            if (kind == Kind::Fold)
            {
                settlement.winner = source.Terminal().foldedPlayer->Other();
                node.utilities[0][0] = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - rootHalfPot;
                node.utilities[1][0] = -node.utilities[0][0] - settlement.rake;
            }
            else
            {
                for (std::size_t outcome = 0; outcome < 3; ++outcome)
                {
                    settlement.winner = outcome == 1 ? std::nullopt : std::optional<core::PlayerId>(core::PlayerId(outcome == 0 ? 0 : 1));
                    node.utilities[0][outcome] = settlement.NetPayoffFromStart(core::PlayerId::Player0()) - rootHalfPot;
                    node.utilities[1][2 - outcome] = -node.utilities[0][outcome] - settlement.rake;
                }
                if (kind == Kind::Showdown)
                    node.rankRow = tables.RankRow(state.board);
                else
                    at.runoutRows = std::max(at.runoutRows, RunoutRow(node) + 1);
            }
        }
        const auto index = at.node++;
        nodes[index] = node;
        at.child += node.childCount;
        at.maxDepth = std::max(at.maxDepth, depth + 1);
        for (std::size_t action = 0; action < node.childCount; ++action)
        {
            const auto child = source.Child(action);
            auto childIndex = at.node;
            if (above && kind == Kind::Chance)
            {
                starts.push_back({at, depth + 1, child});
                at.node += static_cast<std::uint32_t>(child.TraversalNodeCount());
                at.child += child.TraversalNodeCount() - 1;
                // The subtree's decisions of each player with their actions.
                for (std::size_t player = 0; player < 2; ++player)
                {
                    const auto units = decisionUnits(child.ActionEntryCounts()[player], child.DecisionNodeCounts()[player], player);
                    at.strategy += units[0];
                    at.state += units[1];
                }
            }
            else
                childIndex = self(self, child, depth + 1, at, above);
            children[node.childOffset + action] = childIndex;
            if (node.kind == Kind::Chance)
                dealtCards[node.childOffset + action] =
                    static_cast<std::uint8_t>(child.State().board.CardAt(state.board.CardCount()).Index());
        }
        return index;
    };
    Cursor top;
    visit(visit, rootNode, 0, top, true);
    std::vector<Cursor> ends(starts.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t subtree = 0; subtree < static_cast<std::int64_t>(starts.size()); ++subtree)
    {
        ends[subtree] = starts[subtree].at;
        visit(visit, starts[subtree].root, starts[subtree].depth, ends[subtree], false);
    }
    strategySize = top.strategy;
    stateSize = top.state;
    ends.push_back(top);
    for (const auto& end : ends)
    {
        maxActions = std::max(maxActions, end.maxActions);
        maxDepth = std::max(maxDepth, end.maxDepth);
        runoutRows = std::max(runoutRows, end.runoutRows);
    }
    PrepareChanceTasks();
    if (prepareTraining)
        PrepareRunoutOutcomes();
}

void HandTraversalData::PrepareChanceTasks()
{
    VisitChanceGroups(
        tables.game->GetNode(nodes.front().id),
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
    const auto& hands = tables.hands;
    const auto& masks = tables.handMasks;
    const std::size_t pairs = hands[0].size() * hands[1].size();
    runoutOutcomes_.assign(2 * runoutRows * pairs, 0);
    // Each rank row is one unordered pair of undealt cards: a runout of the flop row and,
    // with either card as the turn, a river of that card's turn row.
    for (int high = 0; high < 52; ++high)
        for (int low = 0; low < high; ++low)
        {
            const int row = tables.rowsByRunout[core::CardPairIndex(core::Card(high), core::Card(low))];
            if (row < 0)
                continue;
            std::uint32_t* targets[3] = {runoutOutcomes_.data()};
            std::size_t targetCount = 1;
            for (const int turn : {high, low})
                if (static_cast<std::size_t>(turn) + 1 < runoutRows)
                    targets[targetCount++] = runoutOutcomes_.data() + (static_cast<std::size_t>(turn) + 1) * pairs;
            const auto& ranks = tables.rankRows[row];
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

StrategySnapshot HandTraversalData::ExportStrategy(std::vector<std::uint16_t> sums) const
{
    return ExportStrategy(sums.size(), [&](const std::vector<std::size_t>&, const auto& consume) { consume(sums.data()); });
}

StrategySnapshot HandTraversalData::ExportStrategy(std::size_t chunkUnits, const SumsStream& stream) const
{
    // Blocks follow preorder, whose node IDs increase; each owns its probability range, so
    // decisions normalize independently below.
    std::vector<StrategySnapshot::NodeBlock> blocks;
    std::vector<std::uint32_t> decisions;
    std::vector<core::HoleCards> handLists;
    std::unordered_map<std::uint64_t, std::array<std::size_t, 2>> lists; // offset and count by board mask and actor
    std::size_t probabilityCount = 0;
    for (std::uint32_t index = 0; index < nodes.size(); ++index)
    {
        const Node& node = nodes[index];
        if (node.kind != Kind::Decision)
            continue;
        const auto [list, created] = lists.try_emplace(node.boardMask << 1 | node.actor);
        if (created)
        {
            list->second[0] = handLists.size();
            for (const auto& hand : tables.hands[node.actor])
                if (!(hand.mask & node.boardMask))
                    handLists.push_back(hand.cards);
            list->second[1] = handLists.size() - list->second[0];
        }
        const auto [handOffset, handCount] = list->second;
        if (handCount == 0)
            continue;
        blocks.push_back({node.id, handOffset, probabilityCount, handCount, node.childCount});
        decisions.push_back(index);
        probabilityCount += handCount * node.childCount;
    }
    // Each range starts at its first block's decision, which ends[range - 1] holds.
    std::vector<std::size_t> ends, firstBlocks{0};
    for (std::size_t block = 0, begin = 0; block < blocks.size(); ++block)
    {
        const Node& node = nodes[decisions[block]];
        if (node.stateOffset + StateUnits(node.childCount, tables.hands[node.actor].size()) - begin > chunkUnits &&
            block > firstBlocks.back())
        {
            begin = node.stateOffset;
            ends.push_back(begin);
            firstBlocks.push_back(block);
        }
    }
    ends.push_back(stateSize);
    firstBlocks.push_back(blocks.size());
    std::unique_ptr<float[]> probabilities(new float[probabilityCount]);
    std::size_t range = 0;
    stream(
        ends,
        [&](const std::uint16_t* units)
        {
            const std::size_t begin = range ? ends[range - 1] : 0;
        // Workers write every probability, touching fresh pages in parallel.
#pragma omp parallel for schedule(dynamic, 256)
            for (std::int64_t block = firstBlocks[range]; block < static_cast<std::int64_t>(firstBlocks[range + 1]); ++block)
            {
                const Node& node = nodes[decisions[block]];
                const auto& hands = tables.hands[node.actor];
                float* output = probabilities.get() + blocks[block].probabilityOffset;
                // Action-major sums normalize into the compact hand-major output.
                for (std::size_t hand = 0; hand < hands.size(); ++hand)
                {
                    if (hands[hand].mask & node.boardMask)
                        continue;
                    NormalizeAverageStrategy(units + (node.stateOffset - begin) + hand, RowUnits(hands.size()), node.childCount, output, 1);
                    output += node.childCount;
                }
            }
            ++range;
        }
    );
    return StrategySnapshot(tables.game, std::move(blocks), std::move(handLists), std::move(probabilities));
}

namespace
{
// One buffer of the quantized layout as floats in the unpadded layout, its exponent slots zero.
template<typename Unit>
std::vector<float> DecodeUnits(const HandTraversalData& data, const std::vector<Unit>& units)
{
    std::vector<float> values(data.strategySize);
#pragma omp parallel for schedule(dynamic, 256)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(data.nodes.size()); ++index)
    {
        const auto& node = data.nodes[index];
        if (node.kind != HandTraversalData::Kind::Decision)
            continue;
        const auto hands = data.tables.hands[node.actor].size(), row = HandTraversalData::RowUnits(hands);
        const Unit* nodeUnits = units.data() + node.stateOffset;
        const auto* exponents = ExponentBytes(nodeUnits, node.childCount, row);
        for (std::size_t action = 0; action < node.childCount; ++action)
            for (std::size_t hand = 0; hand < hands; ++hand)
                values[node.strategyOffset + action * hands + hand] =
                    gpu::Dequantize(static_cast<float>(nodeUnits[action * row + hand]), exponents[hand]);
    }
    return values;
}
} // namespace

TrainingState HandTraversalData::Decode(const QuantizedState& state) const
{
    return {DecodeUnits(*this, state.regrets), DecodeUnits(*this, state.strategySums), state.stamps};
}
} // namespace solver::engine
