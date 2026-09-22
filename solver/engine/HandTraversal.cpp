#include "engine/HandTraversal.h"
#include "engine/AverageStrategy.h"
#include "engine/HandEvaluation.h"
#include "engine/ChanceGroups.h"
#include <algorithm>
#include <limits>
#include <utility>
#include <omp.h>

namespace solver::engine
{
HandTraversal::HandTraversal(const SolveProblem& problem, game::NodeId root, bool prepareTraining)
    : HandTraversal(std::make_shared<const HandTraversalData>(problem, root, prepareTraining))
{}

HandTraversal::HandTraversal(std::shared_ptr<const HandTraversalData> data)
    : data_(std::move(data))
    , hands(data_->tables->hands)
    , nodes(data_->nodes)
    , strategySize(data_->strategySize)
    , maxActions(data_->maxActions)
    , rootHalfPot(data_->rootHalfPot)
    , handMasks(data_->tables->handMasks)
    , children(data_->children)
    , dealtCardMasks(data_->dealtCardMasks)
    , rankRows(data_->tables->rankRows)
    , rowsByRunout(data_->tables->rowsByRunout)
    , maxDepth(data_->maxDepth)
    , chanceGroups_(data_->chanceGroups_)
    , chanceTasks_(data_->chanceTasks_)
    , flopOutcomes_(data_->flopOutcomes_)
{}

void HandTraversal::EvaluateFlopRunout(
    const Node& node,
    std::size_t player,
    const float* opponentReach,
    const float* divisors,
    float* values
) const
{
    const float tie = player == 0 ? node.utilities[1] : -node.utilities[1];
    const float win = player == 0 ? node.utilities[0] : -node.utilities[2];
    const float loss = player == 0 ? node.utilities[2] : -node.utilities[0];
    const float winScale = (win - tie) / 990.0f;
    const float lossScale = (loss - tie) / 990.0f;
    std::array<float, kMaxHands> wins{}, losses{};
    // Both orientations read contiguous rows from the same immutable table.
    for (std::size_t first = 0; first < hands[0].size(); ++first)
        for (std::size_t second = 0; second < hands[1].size(); ++second)
        {
            const auto outcomes = flopOutcomes_[first * hands[1].size() + second];
            const auto hand = player == 0 ? first : second;
            const float reach = opponentReach[player == 0 ? second : first];
            wins[hand] += reach * (player == 0 ? outcomes.wins : outcomes.losses);
            losses[hand] += reach * (player == 0 ? outcomes.losses : outcomes.wins);
        }
    const auto masses = tie == 0.0f ? std::vector<float>{} : CompatibleMasses(player, opponentReach);
    for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
    {
        const float baseline = tie == 0.0f ? 0.0f : masses[hand] * tie;
        values[hand] = divisors[hand] > 0.0f ? (baseline + wins[hand] * winScale + losses[hand] * lossScale) / divisors[hand] : 0.0f;
    }
}

std::vector<float> HandTraversal::OpponentReachAtRoot(const StrategySnapshot& strategy, std::size_t opponentPlayer) const
{
    const auto& game = strategy.Game();
    std::vector<game::ParentEdge> path;
    for (auto node = nodes.front().id; game.GetNode(node).Parent();)
    {
        const auto parent = *game.GetNode(node).Parent();
        path.push_back(parent);
        node = parent.node;
    }
    std::vector<float> reach;
    for (const Hand& hand : hands[opponentPlayer])
    {
        float weight = hand.weight;
        for (auto step = path.rbegin(); step != path.rend(); ++step)
        {
            const auto& node = game.GetNode(step->node);
            if (node.Kind() == game::NodeKind::Decision && node.State().playerToAct.Index() == opponentPlayer)
            {
                weight *= strategy.ActionProbability({step->node, hand.cards}, step->edgeIndex);
            }
        }
        // The root board already removes dealt cards. Earlier chance probabilities and
        // the queried hand's own reach cancel in its conditional opponent distribution.
        reach.push_back(weight);
    }
    return reach;
}

std::vector<float> HandTraversal::CompatibleMasses(std::size_t player, const float* opponentReach) const
{
    return CompatibleHandMasses(*data_->tables, player, opponentReach);
}

void HandTraversal::PropagateChild(
    std::uint32_t nodeIndex,
    std::size_t action,
    std::size_t player,
    bool includeChance,
    const float* strategy,
    const float* parent,
    float* child
) const
{
    const Node& node = nodes[nodeIndex];
    const std::size_t count = hands[player].size();
    if (node.kind == game::NodeKind::Chance)
    {
        const float chance = includeChance ? 1.0f / (node.childCount - 4) : 1.0f;
        const auto mask = dealtCardMasks[node.childOffset + action];
        const std::uint64_t* masks = handMasks[player].data();
        for (std::size_t hand = 0; hand < count; ++hand)
            child[hand] = masks[hand] & mask ? 0.0f : parent[hand] * chance;
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
    const float* opponentReach,
    const float* divisors,
    float* values
) const
{
    const auto& tables = *data_->tables;
    if (node.rankRow < 0)
    {
        const float fold = updatingPlayer == 0 ? node.utilities[0] : -node.utilities[0];
        EvaluateFoldHands(tables, updatingPlayer, node.boardMask, opponentReach, divisors, fold, values);
        return;
    }
    const std::array<float, 3> utilities =
        updatingPlayer == 0 ? node.utilities : std::array<float, 3>{-node.utilities[2], -node.utilities[1], -node.utilities[0]};
    EvaluateShowdownHands(tables, updatingPlayer, rankRows[node.rankRow], opponentReach, divisors, utilities, values);
}

void HandTraversal::EvaluateRunout(
    Node node,
    std::size_t player,
    const float* opponentReach,
    const float* divisors,
    float* values,
    bool useCache
) const
{
    if (useCache && node.board.CardCount() == 3 && !flopOutcomes_.empty())
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
    std::array<float, kMaxHands> accumulated{};
    std::array<float, kMaxHands> childReach;
    std::array<float, kMaxHands> childValues;
    const float chance = 1.0f / (52 - node.board.CardCount() - 4);
    const std::uint64_t* masks = handMasks[1 - player].data();
    const std::size_t opponentCount = hands[1 - player].size();
    for (int card = 0; card < 52; ++card)
    {
        const auto mask = std::uint64_t{1} << card;
        if (node.boardMask & mask)
            continue;
        Node child = node;
        child.board = node.board.Append(core::Card(card));
        child.boardMask |= mask;
        for (std::size_t hand = 0; hand < opponentCount; ++hand)
            childReach[hand] = masks[hand] & mask ? 0.0f : opponentReach[hand] * chance;
        EvaluateRunout(child, player, childReach.data(), divisors, childValues.data(), useCache);
        for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
            accumulated[hand] += childValues[hand];
    }
    for (std::size_t hand = 0; hand < hands[player].size(); ++hand)
        values[hand] = accumulated[hand];
}

HandTraversal::WorkspaceSize HandTraversal::SizeWorkspace(
    std::size_t depth,
    std::size_t actions,
    const std::array<std::size_t, 2>& handCounts,
    std::size_t chanceTasks
)
{
    const auto count = std::max(handCounts[0], handCounts[1]);
    return {
        {(depth + 1) * handCounts[0], (depth + 1) * handCounts[1]},
        depth * actions * count,
        depth * count,
        chanceTasks * count,
        depth * actions * count,
    };
}

std::uint64_t HandTraversal::WorkspaceSize::Bytes() const
{
    return (reach[0] + reach[1] + childValues + parallelValues + strategies + accumulated) * sizeof(float);
}

HandTraversal::StorageEstimate HandTraversal::EstimateStorage(
    const game::CompiledGame& game,
    const std::array<std::size_t, 2>& handCounts,
    bool prepareTraining
)
{
    const auto size = game.Size();
    const auto& board = game.Spec().initialBoard;
    const auto totalHands = handCounts[0] + handCounts[1];
    const auto maxHands = std::max(handCounts[0], handCounts[1]);
    std::uint64_t chanceTasks = 0, chancePlanBytes = 0;
    if (prepareTraining)
        VisitChanceGroups(
            game.GetNode(game.Root()),
            [&](const game::GameNode& node, std::uint32_t, const std::vector<std::uint32_t>& path)
            {
                chanceTasks += node.ChanceOutcomeCount();
                chancePlanBytes +=
                    sizeof(ChanceGroup) + path.size() * sizeof(std::uint32_t) + node.ChanceOutcomeCount() * sizeof(ChanceTask);
            }
        );
    const std::uint64_t layout = size.traversalNodes * (sizeof(Node) + sizeof(std::uint32_t) + sizeof(std::uint64_t));
    const std::uint64_t rankBytes = 4 * sizeof(std::uint16_t) + 2 * sizeof(std::uint8_t) + sizeof(std::uint64_t);
    const std::uint64_t undealt = 52 - board.CardCount();
    const std::uint64_t runouts = board.CardCount() == 3 ? undealt * (undealt - 1) / 2 : board.CardCount() == 4 ? undealt : 1;
    const std::uint64_t ranks =
        runouts * (rankBytes * totalHands + 2 * sizeof(RankOrder)) + 2 * totalHands * sizeof(Hand) + totalHands * sizeof(std::uint64_t);
    return {
        layout + ranks + 2 * chancePlanBytes,
        SizeWorkspace(size.depth, size.maxActions, handCounts, 0).Bytes(),
        chanceTasks * maxHands * sizeof(float),
        prepareTraining && board.CardCount() == 3 ? handCounts[0] * handCounts[1] * sizeof(FlopOutcomes) : 0,
    };
}

HandTraversal::Workspace HandTraversal::MakeWorkspace(bool parallel) const
{
    Workspace workspace;
    const auto size = SizeWorkspace(maxDepth, maxActions, {hands[0].size(), hands[1].size()}, parallel ? chanceTasks_.size() : 0);
    for (std::size_t player = 0; player < 2; ++player)
        workspace.reach[player].resize(size.reach[player]);
    workspace.childValues.resize(size.childValues);
    workspace.accumulated.resize(size.accumulated);
    workspace.parallelValues.resize(size.parallelValues);
    workspace.strategies.resize(size.strategies);
    return workspace;
}

void HandTraversal::MatchRegrets(const Node& node, const TrainState& train, float* current) const
{
    const auto count = hands[node.actor].size();
    std::array<float, kMaxHands> positiveRegrets;
    std::fill_n(positiveRegrets.data(), count, 0.0f);
    const float* regrets = train.regrets + node.strategyOffset;
    for (std::size_t action = 0; action < node.childCount; ++action)
    {
        float* row = current + action * count;
        const float* regretRow = regrets + action * count;
        for (std::size_t hand = 0; hand < count; ++hand)
        {
            const float regret = regretRow[hand];
            const float positive = regret > 0.0f ? regret : 0.0f;
            row[hand] = positive;
            positiveRegrets[hand] += positive;
        }
    }
    const float uniform = 1.0f / node.childCount;
    for (std::size_t action = 0; action < node.childCount; ++action)
    {
        float* row = current + action * count;
        for (std::size_t hand = 0; hand < count; ++hand)
            row[hand] = positiveRegrets[hand] > 0.0f ? row[hand] / positiveRegrets[hand] : uniform;
    }
}

void HandTraversal::WalkTraining(
    std::size_t player,
    const float* divisors,
    Workspace& workspace,
    float* values,
    std::vector<Workspace>& workers,
    TrainState& train
) const
{
    WalkContext context{player, divisors};
    context.train = &train;
    context.useRunoutCache = true;
    if (workers.size() <= 1 || chanceTasks_.empty())
    {
        Walk(0, context, workspace, 0, values);
        return;
    }
    const auto count = hands[player].size();
    const int team = static_cast<int>(std::min(workers.size(), chanceTasks_.size()));
    // Ancestor regrets stay unchanged until all disjoint subtrees finish. Replaying
    // their short paths avoids retaining separate strategy/reach snapshots.
#pragma omp parallel for num_threads(team) schedule(dynamic, 1)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(chanceTasks_.size()); ++index)
    {
        const ChanceTask& task = chanceTasks_[index];
        const ChanceGroup& group = chanceGroups_[task.group];
        Workspace& scratch = workers[omp_get_thread_num()];
        for (std::size_t p = 0; p < 2; ++p)
            std::copy_n(workspace.reach[p].data(), hands[p].size(), scratch.reach[p].data());
        std::uint32_t nodeIndex = 0;
        std::size_t depth = 0;
        for (const auto action : group.path)
        {
            const Node& node = nodes[nodeIndex];
            float* policy = scratch.strategies.data();
            MatchRegrets(node, train, policy);
            for (std::size_t p = 0; p < 2; ++p)
                PropagateChild(
                    nodeIndex,
                    action,
                    p,
                    p != player,
                    policy,
                    scratch.reach[p].data() + depth * hands[p].size(),
                    scratch.reach[p].data() + (depth + 1) * hands[p].size()
                );
            nodeIndex = children[node.childOffset + action];
            ++depth;
        }
        for (std::size_t p = 0; p < 2; ++p)
            PropagateChild(
                group.node,
                task.action,
                p,
                p != player,
                nullptr,
                scratch.reach[p].data() + depth * hands[p].size(),
                scratch.reach[p].data() + (depth + 1) * hands[p].size()
            );
        Walk(
            children[nodes[group.node].childOffset + task.action],
            context,
            scratch,
            depth + 1,
            workspace.parallelValues.data() + index * count
        );
    }
    // Consume each result once in the same preorder/action order as serial Walk.
    std::size_t cursor = 0;
    Walk(0, context, workspace, 0, values, &cursor);
}

std::vector<float> HandTraversal::EvaluateSnapshot(
    const StrategySnapshot& strategy,
    std::size_t player,
    const std::vector<float>& opponentReach,
    const std::vector<float>& divisors,
    Evaluation evaluation
) const
{
    WalkContext context{player, divisors.data()};
    context.strategy = &strategy;
    context.bestResponse = evaluation == Evaluation::BestResponse;
    return EvaluateHands(context, opponentReach);
}

std::vector<float> HandTraversal::EvaluateAverageBestResponse(
    const float* strategySums,
    std::size_t player,
    const std::vector<float>& opponentReach,
    const std::vector<float>& divisors
) const
{
    WalkContext context{player, divisors.data()};
    context.strategySums = strategySums;
    context.bestResponse = true;
    return EvaluateHands(context, opponentReach);
}

std::vector<float> HandTraversal::EvaluateHands(const WalkContext& context, const std::vector<float>& opponentReach) const
{
    auto workspace = MakeWorkspace();
    std::copy(opponentReach.begin(), opponentReach.end(), workspace.reach[1 - context.player].begin());
    std::vector<float> values(hands[context.player].size());
    Walk(0, context, workspace, 0, values.data());
    return values;
}

void HandTraversal::Walk(
    std::uint32_t nodeIndex,
    const WalkContext& context,
    Workspace& workspace,
    std::size_t depth,
    float* values,
    std::size_t* parallelCursor
) const
{
    const auto player = context.player;
    const auto* divisors = context.divisors;
    const auto* strategy = context.strategy;
    const auto* strategySums = context.strategySums;
    auto* train = context.train;
    const bool bestResponse = context.bestResponse;
    const Node& node = nodes[nodeIndex];
    const auto count = hands[player].size();
    const auto stride = std::max(hands[0].size(), hands[1].size());
    const auto opponentCount = hands[1 - player].size();
    const float* opponentReach = workspace.reach[1 - player].data() + depth * opponentCount;
    if (node.forcedRunout)
    {
        // Checkpoints use the same runout accumulation as exported-snapshot evaluation.
        EvaluateRunout(node, player, opponentReach, divisors, values, context.useRunoutCache);
        return;
    }
    if (node.kind == game::NodeKind::Terminal)
    {
        EvaluateTerminal(node, player, opponentReach, divisors, values);
        return;
    }
    const float* nodeStrategy = nullptr;
    // Best response maximizes own action values and propagates only opponent reach.
    if (node.kind == game::NodeKind::Decision && !(bestResponse && node.actor == player))
    {
        const auto actorCount = hands[node.actor].size();
        float* current = workspace.strategies.data() + depth * maxActions * stride;
        if (train)
        {
            // Preserve this entry strategy until reach, backup and average-strategy
            // accumulation finish. Descendants and other workers use separate rows.
            MatchRegrets(node, *train, current);
        }
        else if (strategySums)
        {
            const float* sums = strategySums + node.strategyOffset;
            for (std::size_t hand = 0; hand < actorCount; ++hand)
                NormalizeAverageStrategy(sums + hand, actorCount, node.childCount, current + hand, actorCount);
        }
        else
        {
            const auto source = strategy->FindNodeStrategy(node.id);
            const float uniform = 1.0f / node.childCount;
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
    const bool parallel = node.kind == game::NodeKind::Chance && parallelCursor;
    float* childrenValues =
        parallel ? workspace.parallelValues.data() + *parallelCursor * count : workspace.childValues.data() + depth * maxActions * stride;
    if (parallel)
        *parallelCursor += node.childCount;
    float* accumulated = workspace.accumulated.data() + depth * stride;
    std::fill_n(accumulated, count, acting && bestResponse ? -std::numeric_limits<float>::infinity() : 0.0f);
    const auto descend = [&](std::size_t action, float* output)
    {
        const auto childIndex = children[node.childOffset + action];
        const Node& childNode = nodes[childIndex];
        // Leaves consume only opponent reach; own reach is needed for strategy updates.
        const bool needsOwnReach = train && childNode.kind != game::NodeKind::Terminal && !childNode.forcedRunout;
        for (std::size_t p = 0; p < 2; ++p)
            if (p != player || needsOwnReach)
                PropagateChild(
                    nodeIndex,
                    action,
                    p,
                    p != player,
                    nodeStrategy,
                    workspace.reach[p].data() + depth * hands[p].size(),
                    workspace.reach[p].data() + (depth + 1) * hands[p].size()
                );
        Walk(childIndex, context, workspace, depth + 1, output, parallelCursor);
    };
    for (std::size_t action = 0; action < node.childCount; ++action)
    {
        // Serial chance branches reuse one row; decision rows survive until the regret update.
        float* child = childrenValues + ((parallel || node.kind == game::NodeKind::Decision) ? action * count : 0);
        if (!parallel)
            descend(action, child);
        if (acting && bestResponse)
            for (std::size_t hand = 0; hand < count; ++hand)
                accumulated[hand] = std::max(accumulated[hand], child[hand]);
        else if (acting)
        {
            const float* probability = nodeStrategy + action * count;
            for (std::size_t hand = 0; hand < count; ++hand)
                accumulated[hand] += probability[hand] * child[hand];
        }
        else
            for (std::size_t hand = 0; hand < count; ++hand)
                accumulated[hand] += child[hand];
    }
    for (std::size_t hand = 0; hand < count; ++hand)
        values[hand] = accumulated[hand];
    if (acting && train)
    {
        const float* ownReach = workspace.reach[player].data() + depth * count;
        const float positiveDiscount = train->positiveDiscount;
        const float averageDiscount = train->averageDiscount;
        float* regrets = train->regrets + node.strategyOffset;
        float* strategySums = train->strategySums + node.strategyOffset;
        for (std::size_t action = 0; action < node.childCount; ++action)
        {
            const float* childValues = childrenValues + action * count;
            const float* probabilities = nodeStrategy + action * count;
            float* regretRow = regrets + action * count;
            float* sumRow = strategySums + action * count;
            for (std::size_t hand = 0; hand < count; ++hand)
            {
                const float regret = regretRow[hand] + (childValues[hand] - values[hand]);
                regretRow[hand] = regret * (regret > 0.0f ? positiveDiscount : 0.5f);
                sumRow[hand] = averageDiscount * (sumRow[hand] + ownReach[hand] * probabilities[hand]);
            }
        }
    }
}
} // namespace solver::engine
