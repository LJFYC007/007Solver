#include "engine/HandTraversal.h"
#include "engine/AverageStrategy.h"
#include "engine/HandEvaluation.h"
#include "engine/ChanceGroups.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <omp.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace solver::engine
{
namespace
{
// Win/tie/loss payoffs for player at a showdown or forced runout.
std::array<float, 3> ShowdownUtilities(const HandTraversalData::Node& node, std::size_t player)
{
    return player == 0 ? node.utilities : std::array<float, 3>{-node.utilities[2], -node.utilities[1], -node.utilities[0]};
}

// Adds weight * increments to sums, skipping 8-wide chunks that are all zero, which is
// most entries once play prunes lines; the skipped update is exact. MSVC does not
// vectorize the chunk test, so AVX2 builds spell it out.
void AccumulateStrategy(float* sums, const float* increments, std::size_t count, float weight)
{
    const std::size_t aligned = count / 8 * 8;
#if defined(__AVX2__)
    const __m256 scale = _mm256_set1_ps(weight);
    for (std::size_t chunk = 0; chunk < aligned; chunk += 8)
    {
        const __m256 values = _mm256_loadu_ps(increments + chunk);
        if (_mm256_movemask_ps(_mm256_cmp_ps(values, _mm256_setzero_ps(), _CMP_NEQ_UQ)) == 0)
            continue;
        _mm256_storeu_ps(sums + chunk, _mm256_add_ps(_mm256_loadu_ps(sums + chunk), _mm256_mul_ps(scale, values)));
    }
#else
    for (std::size_t chunk = 0; chunk < aligned; chunk += 8)
    {
        bool any = false;
        for (int i = 0; i < 8; ++i)
            any |= increments[chunk + i] != 0.0f;
        if (!any)
            continue;
        for (int i = 0; i < 8; ++i)
            sums[chunk + i] += weight * increments[chunk + i];
    }
#endif
    for (std::size_t hand = aligned; hand < count; ++hand)
        if (increments[hand] != 0.0f)
            sums[hand] += weight * increments[hand];
}
} // namespace

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
    , children(data_->children)
    , dealtCards(data_->dealtCards)
    , rankRows(data_->tables->rankRows)
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
    const auto [win, tie, loss] = ShowdownUtilities(node, player);
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
    // The opponent's decisions from this root up to the game root.
    std::vector<game::ParentEdge> path;
    for (auto node = game.GetNode(nodes.front().id); node.Parent();)
    {
        const auto parent = *node.Parent();
        node = game.GetNode(parent.node);
        if (node.Kind() == game::NodeKind::Decision && node.State().playerToAct.Index() == opponentPlayer)
            path.push_back(parent);
    }
    std::vector<float> reach;
    for (const Hand& hand : hands[opponentPlayer])
    {
        float weight = hand.weight;
        for (auto step = path.rbegin(); step != path.rend(); ++step)
            weight *= strategy.ActionProbability({step->node, hand.cards}, step->edgeIndex);
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

const float* HandTraversal::PropagateChild(
    std::uint32_t nodeIndex,
    std::size_t action,
    std::size_t player,
    const float* strategy,
    const float* parent,
    float* child
) const
{
    const Node& node = nodes[nodeIndex];
    if (node.kind == Kind::Decision && node.actor != player)
        return parent;
    const std::size_t count = hands[player].size();
    if (node.kind == Kind::Chance)
    {
        // Multiplying by an exact zero/one factor keeps the loop branch-free and vectorizable.
        const float chance = 1.0f / (node.childCount - 4);
        const float* factors = data_->tables->cardFactors[player].data() + std::size_t(dealtCards[node.childOffset + action]) * count;
        for (std::size_t hand = 0; hand < count; ++hand)
            child[hand] = parent[hand] * factors[hand] * chance;
    }
    else
    {
        const float* probabilities = strategy + action * count;
        for (std::size_t hand = 0; hand < count; ++hand)
            child[hand] = parent[hand] * probabilities[hand];
    }
    return child;
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
    if (node.kind == Kind::Fold)
    {
        const float fold = updatingPlayer == 0 ? node.utilities[0] : -node.utilities[0];
        EvaluateFoldHands(tables, updatingPlayer, node.boardMask, opponentReach, divisors, fold, values);
        return;
    }
    EvaluateShowdownHands(
        tables, updatingPlayer, rankRows[node.rankRow], opponentReach, divisors, ShowdownUtilities(node, updatingPlayer), values
    );
}

void HandTraversal::EvaluateRunout(
    const std::array<float, 3>& utilities,
    const core::Board& board,
    std::uint64_t boardMask,
    std::size_t player,
    const float* opponentReach,
    const float* divisors,
    float* values
) const
{
    const auto& tables = *data_->tables;
    if (board.CardCount() == 5)
    {
        EvaluateShowdownHands(tables, player, rankRows[tables.RankRow(board)], opponentReach, divisors, utilities, values);
        return;
    }
    std::array<float, kMaxHands> accumulated{};
    std::array<float, kMaxHands> childReach;
    std::array<float, kMaxHands> childValues;
    const float chance = 1.0f / (52 - board.CardCount() - 4);
    const std::size_t opponentCount = hands[1 - player].size();
    for (int card = 0; card < 52; ++card)
    {
        const auto mask = std::uint64_t{1} << card;
        if (boardMask & mask)
            continue;
        const float* factors = tables.cardFactors[1 - player].data() + std::size_t(card) * opponentCount;
        for (std::size_t hand = 0; hand < opponentCount; ++hand)
            childReach[hand] = opponentReach[hand] * factors[hand] * chance;
        EvaluateRunout(
            utilities, board.Append(core::Card(card)), boardMask | mask, player, childReach.data(), divisors, childValues.data()
        );
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
        (depth + 1) * count,
        depth * actions * count,
        depth * count,
        chanceTasks * count,
        depth * actions * count,
    };
}

std::uint64_t HandTraversal::WorkspaceSize::Bytes() const
{
    return (reach + childValues + parallelValues + strategies + accumulated) * sizeof(float);
}

HandTraversal::StorageEstimate HandTraversal::EstimateStorage(
    const game::CompiledGame& game,
    const std::array<std::size_t, 2>& handCounts,
    bool prepareTraining
)
{
    const auto size = game.Size();
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
    const std::uint64_t layout = size.traversalNodes * (sizeof(Node) + sizeof(std::uint32_t) + sizeof(std::uint8_t));
    const std::uint64_t rankBytes = 6 * sizeof(std::uint16_t) + 3 * sizeof(std::uint32_t);
    // Games start on the flop, so every turn/river pair has a rank row.
    const std::uint64_t undealt = 52 - game.Spec().initialBoard.CardCount();
    const std::uint64_t runouts = undealt * (undealt - 1) / 2;
    const std::uint64_t ranks = runouts * (rankBytes * totalHands + 2 * sizeof(RankOrder)) + 2 * totalHands * sizeof(Hand) +
                                totalHands * (sizeof(std::uint64_t) + 52 * sizeof(float));
    return {
        layout + ranks + 2 * chancePlanBytes,
        SizeWorkspace(size.depth, size.maxActions, handCounts, 0).Bytes(),
        chanceTasks * (maxHands * sizeof(float) + sizeof(std::uint8_t)),
        prepareTraining ? handCounts[0] * handCounts[1] * sizeof(FlopOutcomes) : 0,
    };
}

HandTraversal::Workspace HandTraversal::MakeWorkspace(bool parallel) const
{
    Workspace workspace;
    const auto size = SizeWorkspace(maxDepth, maxActions, {hands[0].size(), hands[1].size()}, parallel ? chanceTasks_.size() : 0);
    workspace.reach.resize(size.reach);
    workspace.childValues.resize(size.childValues);
    workspace.accumulated.resize(size.accumulated);
    workspace.parallelValues.resize(size.parallelValues);
    workspace.parallelLive.resize(parallel ? chanceTasks_.size() : 0);
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
    const float* rootReach = workspace.reach.data();
    if (workers.size() <= 1 || chanceTasks_.empty())
    {
        Walk(0, context, workspace, 0, rootReach, values);
        return;
    }
    const auto count = hands[player].size();
    const auto stride = std::max(hands[0].size(), hands[1].size());
    const auto opponent = 1 - player;
    const int team = static_cast<int>(std::min(workers.size(), chanceTasks_.size()));
    // Ancestor regrets stay unchanged until all disjoint subtrees finish. Replaying
    // their short paths avoids retaining separate strategy/reach snapshots; the serial
    // walk below accumulates the ancestors' strategy sums. Workers read the root reach
    // row in place and write reach only to their own scratch.
#pragma omp parallel for num_threads(team) schedule(dynamic, 1)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(chanceTasks_.size()); ++index)
    {
        const ChanceTask& task = chanceTasks_[index];
        const ChanceGroup& group = chanceGroups_[task.group];
        Workspace& scratch = workers[omp_get_thread_num()];
        const float* reach = rootReach;
        std::uint32_t nodeIndex = 0;
        std::size_t depth = 0;
        for (const auto action : group.path)
        {
            const Node& node = nodes[nodeIndex];
            float* policy = scratch.strategies.data();
            if (node.kind == Kind::Decision && node.actor == opponent)
                MatchRegrets(node, train, policy);
            reach = PropagateChild(nodeIndex, action, opponent, policy, reach, scratch.reach.data() + (depth + 1) * stride);
            nodeIndex = children[node.childOffset + action];
            ++depth;
        }
        reach = PropagateChild(group.node, task.action, opponent, nullptr, reach, scratch.reach.data() + (depth + 1) * stride);
        workspace.parallelLive[index] = Walk(
            children[nodes[group.node].childOffset + task.action],
            context,
            scratch,
            depth + 1,
            reach,
            workspace.parallelValues.data() + index * count
        );
    }
    // Consume each result once in the same preorder/action order as serial Walk.
    std::size_t cursor = 0;
    Walk(0, context, workspace, 0, rootReach, values, &cursor);
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
    std::vector<float> values(hands[context.player].size());
    Walk(0, context, workspace, 0, opponentReach.data(), values.data());
    return values;
}

bool HandTraversal::Walk(
    std::uint32_t nodeIndex,
    const WalkContext& context,
    Workspace& workspace,
    std::size_t depth,
    const float* opponentReach,
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
    // Subtrees the opponent never reaches have zero values and zero regret and strategy
    // increments, so they are skipped; the stamps discount their regrets lazily. The walk
    // consuming completed chance tasks keeps descending so its cursor stays in preorder.
    if ((!parallelCursor || node.IsLeaf()) && std::all_of(opponentReach, opponentReach + opponentCount, [](float r) { return r == 0.0f; }))
    {
        std::fill_n(values, count, 0.0f);
        return false;
    }
    if (node.IsLeaf())
    {
        if (node.kind != Kind::ForcedRunout)
            EvaluateTerminal(node, player, opponentReach, divisors, values);
        else if (train && node.board.CardCount() == 3 && !flopOutcomes_.empty())
            EvaluateFlopRunout(node, player, opponentReach, divisors, values);
        else
            EvaluateRunout(ShowdownUtilities(node, player), node.board, node.boardMask, player, opponentReach, divisors, values);
        return true;
    }
    const float* nodeStrategy = nullptr;
    // Best response maximizes own action values and propagates only opponent reach.
    if (node.kind == Kind::Decision && !(bestResponse && node.actor == player))
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
    const bool acting = node.kind == Kind::Decision && node.actor == player;
    const bool parallel = node.kind == Kind::Chance && parallelCursor;
    const std::size_t firstTask = parallel ? *parallelCursor : 0;
    float* childrenValues =
        parallel ? workspace.parallelValues.data() + firstTask * count : workspace.childValues.data() + depth * maxActions * stride;
    if (parallel)
        *parallelCursor += node.childCount;
    float* accumulated = workspace.accumulated.data() + depth * stride;
    std::fill_n(accumulated, count, acting && bestResponse ? -std::numeric_limits<float>::infinity() : 0.0f);
    // The opponent's decisions accumulate its cumulative strategy from the reach they propagate.
    const bool accumulate = train && node.kind == Kind::Decision && !acting;
    const auto descend = [&](std::size_t action, float* output)
    {
        const auto childIndex = children[node.childOffset + action];
        const float* childReach =
            PropagateChild(nodeIndex, action, 1 - player, nodeStrategy, opponentReach, workspace.reach.data() + (depth + 1) * stride);
        if (accumulate)
            AccumulateStrategy(
                train->strategySums + node.strategyOffset + action * opponentCount, childReach, opponentCount, train->weights.averageWeight
            );
        return Walk(childIndex, context, workspace, depth + 1, childReach, output, parallelCursor);
    };
    bool live = false;
    for (std::size_t action = 0; action < node.childCount; ++action)
    {
        // Serial chance branches reuse one row; decision rows survive until the regret update.
        float* child = childrenValues + ((parallel || node.kind == Kind::Decision) ? action * count : 0);
        live |= parallel ? workspace.parallelLive[firstTask + action] != 0 : descend(action, child);
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
    if (acting && train && live)
    {
        const auto& weights = train->weights;
        // Positive regrets are stored divided by positiveScale, so the discounts of the
        // updates this node skipped are already in place; negative regrets halve once per
        // skipped update.
        const std::uint32_t skipped = weights.update - 1 - train->stamps[nodeIndex];
        const float halving = std::ldexp(1.0f, -static_cast<int>(std::min<std::uint32_t>(skipped, 200)));
        const float scale = weights.positiveScale, inverse = weights.positiveInverse;
        float* regrets = train->regrets + node.strategyOffset;
        for (std::size_t action = 0; action < node.childCount; ++action)
        {
            const float* childValues = childrenValues + action * count;
            float* regretRow = regrets + action * count;
            for (std::size_t hand = 0; hand < count; ++hand)
            {
                const float old = regretRow[hand];
                const float regret = (old > 0.0f ? old * scale : old * halving) + (childValues[hand] - values[hand]);
                regretRow[hand] = regret > 0.0f ? regret * inverse : regret * 0.5f;
            }
        }
        train->stamps[nodeIndex] = weights.update;
    }
    return live;
}
} // namespace solver::engine
