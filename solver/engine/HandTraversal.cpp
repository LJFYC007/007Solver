#include "engine/HandTraversal.h"
#include "engine/AverageStrategy.h"
#include "engine/HandEvaluation.h"
#include "engine/ChanceGroups.h"
#include "engine/gpu/GpuQuantize.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
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

#if defined(__AVX2__)
// Eight-hand vectors of GpuQuantize.h's arithmetic, matching its scalar results.
namespace vector
{
inline __m256i LoadExponents(const std::uint8_t* bytes)
{
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(bytes)));
}
inline __m256 PowerOfTwo(__m256i exponent)
{
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(127)), 23));
}
// 2^(byte - bias) and its inverse.
inline __m256 Scale(__m256i exponentBytes)
{
    return PowerOfTwo(_mm256_sub_epi32(exponentBytes, _mm256_set1_epi32(gpu::kExponentBias)));
}
inline __m256 InverseScale(__m256i exponentBytes)
{
    return PowerOfTwo(_mm256_sub_epi32(_mm256_set1_epi32(gpu::kExponentBias), exponentBytes));
}
// Eight 16-bit integers as floats.
inline __m256 Load(const std::int16_t* row)
{
    return _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row))));
}
inline __m256 Load(const std::uint16_t* row)
{
    return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row))));
}
inline __m256i ExponentByte(__m256 magnitude, int bits)
{
    const __m256i biased = _mm256_and_si256(_mm256_srli_epi32(_mm256_castps_si256(magnitude), 23), _mm256_set1_epi32(0xff));
    __m256i exponent = _mm256_max_epi32(_mm256_sub_epi32(biased, _mm256_set1_epi32(126 + bits)), _mm256_set1_epi32(-126));
    const __m256 scaled = _mm256_mul_ps(magnitude, PowerOfTwo(_mm256_sub_epi32(_mm256_setzero_si256(), exponent)));
    // A magnitude beyond the limit takes the next exponent (the compare is all ones there).
    const __m256 beyond = _mm256_cmp_ps(scaled, _mm256_set1_ps(static_cast<float>((1 << bits) - 1)), _CMP_GT_OQ);
    exponent = _mm256_sub_epi32(exponent, _mm256_castps_si256(beyond));
    return _mm256_add_epi32(exponent, _mm256_set1_epi32(gpu::kExponentBias));
}
// The dithers of eight consecutive hands from index at an update (GpuQuantize.h's Dither).
inline __m256 Dither(std::size_t index, std::uint32_t update)
{
    const __m256i lanes =
        _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(static_cast<std::uint32_t>(index))), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256i hash = _mm256_xor_si256(lanes, _mm256_set1_epi32(static_cast<int>(update * gpu::kDitherSpread)));
    hash = _mm256_xor_si256(hash, _mm256_srli_epi32(hash, 16));
    hash = _mm256_mullo_epi32(hash, _mm256_set1_epi32(static_cast<int>(gpu::kDitherMix1)));
    hash = _mm256_xor_si256(hash, _mm256_srli_epi32(hash, 13));
    hash = _mm256_mullo_epi32(hash, _mm256_set1_epi32(static_cast<int>(gpu::kDitherMix2)));
    hash = _mm256_xor_si256(hash, _mm256_srli_epi32(hash, 16));
    return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(hash, 8)), _mm256_set1_ps(1.0f / 16777216.0f));
}
inline __m256i Quantize(__m256 values, __m256 inverseScale, __m256 dither)
{
    const __m256 scaled = _mm256_mul_ps(values, inverseScale);
    const __m256i base = _mm256_cvttps_epi32(scaled);
    const __m256 fraction = _mm256_sub_ps(scaled, _mm256_cvtepi32_ps(base));
    // fraction < -dither is the scalar dither < -fraction with the negation hoisted out of
    // the callers' loops.
    const __m256 roundUp = _mm256_cmp_ps(fraction, _mm256_sub_ps(_mm256_set1_ps(1.0f), dither), _CMP_GE_OQ);
    const __m256 roundDown = _mm256_cmp_ps(fraction, _mm256_sub_ps(_mm256_setzero_ps(), dither), _CMP_LT_OQ);
    return _mm256_add_epi32(_mm256_sub_epi32(base, _mm256_castps_si256(roundUp)), _mm256_castps_si256(roundDown));
}
// Eight 32-bit integers packed to 16 bits; the encoded ranges never reach the saturation.
inline __m128i Pack(__m256i values, bool unsignedRange)
{
    const __m128i low = _mm256_castsi256_si128(values), high = _mm256_extracti128_si256(values, 1);
    return unsignedRange ? _mm_packus_epi32(low, high) : _mm_packs_epi32(low, high);
}
inline void StoreExponents(std::uint8_t* bytes, __m256i exponentBytes)
{
    const __m128i words = Pack(exponentBytes, false);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(bytes), _mm_packus_epi16(words, words));
}
inline __m256 Abs(__m256 values)
{
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), values);
}
// Re-encodes eight hands of per-action rows of count units from the float rows at the
// same positions, at the scales of the hands' magnitudes.
template<typename Unit>
inline void Encode(
    Unit* units,
    std::uint8_t* exponents,
    const float* rows,
    std::size_t actions,
    std::size_t count,
    __m256 magnitude,
    std::size_t index,
    std::uint32_t update
)
{
    constexpr bool isSigned = std::is_signed_v<Unit>;
    const __m256i next = ExponentByte(magnitude, isSigned ? gpu::kRegretBits : gpu::kSumBits);
    const __m256 inverse = InverseScale(next), dither = Dither(index, update);
    for (std::size_t action = 0; action < actions; ++action)
    {
        const __m128i packed = Pack(Quantize(_mm256_loadu_ps(rows + action * count), inverse, dither), !isSigned);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(units + action * count), packed);
    }
    StoreExponents(exponents, next);
}
} // namespace vector
#endif

// Re-encodes one hand of per-action rows of count units from the float rows at the same
// positions, at the scale of its magnitude.
template<typename Unit>
void Encode(
    Unit* units,
    std::uint8_t* exponent,
    const float* rows,
    std::size_t actions,
    std::size_t count,
    float magnitude,
    std::size_t index,
    std::uint32_t update
)
{
    const auto next = gpu::ExponentByte(magnitude, std::is_signed_v<Unit> ? gpu::kRegretBits : gpu::kSumBits);
    const float dither = gpu::Dither(static_cast<std::uint32_t>(index), update);
    for (std::size_t action = 0; action < actions; ++action)
        units[action * count] = static_cast<Unit>(gpu::Quantize(rows[action * count], next, dither));
    *exponent = static_cast<std::uint8_t>(next);
}

// Adds weight * reach * policy to the cumulative strategy of one decision at layout
// offset (action-major rows of count hands), as the GPU's Reach does, re-encoding every
// hand with reach (see GpuQuantize.h) through rows (actions * count floats of scratch).
// Hands without reach keep their entries, so the skipped update is exact; AVX2 builds
// skip eight-hand chunks without reach and re-encode a whole chunk otherwise, which
// keeps the values of its hands without reach exactly.
void AccumulateNodeStrategy(
    std::uint16_t* sums,
    std::size_t offset,
    std::size_t actions,
    std::size_t count,
    const float* reach,
    const float* policy,
    float* rows,
    float weight,
    std::uint32_t update
)
{
    std::uint8_t* exponents = ExponentBytes(sums, actions, count);
    std::size_t hand = 0;
#if defined(__AVX2__)
    const __m256 weights = _mm256_set1_ps(weight);
    for (; hand + 8 <= count; hand += 8)
    {
        const __m256 reaches = _mm256_loadu_ps(reach + hand);
        if (_mm256_movemask_ps(_mm256_cmp_ps(reaches, _mm256_setzero_ps(), _CMP_NEQ_UQ)) == 0)
            continue;
        const __m256 scale = vector::Scale(vector::LoadExponents(exponents + hand));
        __m256 magnitude = _mm256_setzero_ps();
        for (std::size_t action = 0; action < actions; ++action)
        {
            const std::size_t i = action * count + hand;
            const __m256 increment = _mm256_mul_ps(weights, _mm256_mul_ps(reaches, _mm256_loadu_ps(policy + i)));
            const __m256 updated = _mm256_add_ps(_mm256_mul_ps(vector::Load(sums + i), scale), increment);
            _mm256_storeu_ps(rows + i, updated);
            magnitude = _mm256_max_ps(magnitude, updated);
        }
        vector::Encode(sums + hand, exponents + hand, rows + hand, actions, count, magnitude, offset + hand, update);
    }
#endif
    for (; hand < count; ++hand)
    {
        // Locals, because stores to rows may alias the inputs as far as the compiler knows.
        const float handReach = reach[hand];
        if (handReach == 0.0f)
            continue;
        const unsigned int exponent = exponents[hand];
        float magnitude = 0.0f;
        for (std::size_t action = 0; action < actions; ++action)
        {
            const std::size_t i = action * count + hand;
            rows[i] = gpu::Dequantize(static_cast<float>(sums[i]), exponent) + weight * (handReach * policy[i]);
            magnitude = std::max(magnitude, rows[i]);
        }
        Encode(sums + hand, exponents + hand, rows + hand, actions, count, magnitude, offset + hand, update);
    }
}

// The regret update of one acting decision at layout offset from its child value rows
// (actions * count floats, overwritten with the stored regrets) and its values. Positive
// regrets are stored divided by positiveScale, so the discounts of the updates the node
// skipped are already in place; negative regrets halve once per skipped update. Each
// hand is then re-encoded (see GpuQuantize.h).
void UpdateNodeRegrets(
    std::int16_t* regrets,
    std::size_t offset,
    std::size_t actions,
    std::size_t count,
    float* rows,
    const float* values,
    float scale,
    float inverse,
    float halving,
    std::uint32_t update
)
{
    std::uint8_t* exponents = ExponentBytes(regrets, actions, count);
    std::size_t hand = 0;
#if defined(__AVX2__)
    const __m256 zero = _mm256_setzero_ps(), half = _mm256_set1_ps(0.5f);
    const __m256 scales = _mm256_set1_ps(scale), inverses = _mm256_set1_ps(inverse), halvings = _mm256_set1_ps(halving);
    for (; hand + 8 <= count; hand += 8)
    {
        const __m256 value = _mm256_loadu_ps(values + hand);
        const __m256 decode = vector::Scale(vector::LoadExponents(exponents + hand));
        __m256 magnitude = zero;
        for (std::size_t action = 0; action < actions; ++action)
        {
            const std::size_t i = action * count + hand;
            const __m256 old = _mm256_mul_ps(vector::Load(regrets + i), decode);
            const __m256 discounted =
                _mm256_blendv_ps(_mm256_mul_ps(old, halvings), _mm256_mul_ps(old, scales), _mm256_cmp_ps(old, zero, _CMP_GT_OQ));
            const __m256 regret = _mm256_add_ps(discounted, _mm256_sub_ps(_mm256_loadu_ps(rows + i), value));
            const __m256 stored =
                _mm256_blendv_ps(_mm256_mul_ps(regret, half), _mm256_mul_ps(regret, inverses), _mm256_cmp_ps(regret, zero, _CMP_GT_OQ));
            _mm256_storeu_ps(rows + i, stored);
            magnitude = _mm256_max_ps(magnitude, vector::Abs(stored));
        }
        vector::Encode(regrets + hand, exponents + hand, rows + hand, actions, count, magnitude, offset + hand, update);
    }
#endif
    for (; hand < count; ++hand)
    {
        // Locals, because stores to rows may alias the inputs as far as the compiler knows.
        const unsigned int exponent = exponents[hand];
        const float value = values[hand];
        float magnitude = 0.0f;
        for (std::size_t action = 0; action < actions; ++action)
        {
            const std::size_t i = action * count + hand;
            const float old = gpu::Dequantize(static_cast<float>(regrets[i]), exponent);
            const float regret = (old > 0.0f ? old * scale : old * halving) + (rows[i] - value);
            rows[i] = regret > 0.0f ? regret * inverse : regret * 0.5f;
            magnitude = std::max(magnitude, std::fabs(rows[i]));
        }
        Encode(regrets + hand, exponents + hand, rows + hand, actions, count, magnitude, offset + hand, update);
    }
}

// Regret matching for hands [begin, end) of per-action rows of count hands, on the
// quantized regrets (see GpuQuantize.h).
void MatchHands(
    const std::int16_t* regrets,
    float* current,
    std::size_t actions,
    std::size_t count,
    float uniform,
    std::size_t begin,
    std::size_t end
)
{
    std::array<float, HandTraversalData::kMaxHands> positiveRegrets;
    std::fill_n(positiveRegrets.data() + begin, end - begin, 0.0f);
    for (std::size_t action = 0; action < actions; ++action)
    {
        float* row = current + action * count;
        const std::int16_t* regretRow = regrets + action * count;
        for (std::size_t hand = begin; hand < end; ++hand)
        {
            const float regret = static_cast<float>(regretRow[hand]);
            const float positive = regret > 0.0f ? regret : 0.0f;
            row[hand] = positive;
            positiveRegrets[hand] += positive;
        }
    }
    for (std::size_t action = 0; action < actions; ++action)
    {
        float* row = current + action * count;
        for (std::size_t hand = begin; hand < end; ++hand)
            row[hand] = positiveRegrets[hand] > 0.0f ? row[hand] / positiveRegrets[hand] : uniform;
    }
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
    , runoutOutcomes_(data_->runoutOutcomes_)
{}

void HandTraversal::EvaluateRunoutOutcomes(
    const Node& node,
    std::size_t player,
    const float* opponentReach,
    const float* divisors,
    float* values
) const
{
    const auto [win, tie, loss] = ShowdownUtilities(node, player);
    const std::size_t row = HandTraversalData::RunoutRow(node);
    const float runouts = row == 0 ? 990.0f : 44.0f;
    const float winScale = (win - tie) / runouts;
    const float lossScale = (loss - tie) / runouts;
    const std::size_t count = hands[player].size(), opponentCount = hands[1 - player].size();
    const std::size_t pairs = hands[0].size() * hands[1].size();
    // This player's layout of the row keeps its hands contiguous per opponent hand, so the
    // inner loop vectorizes; player 1's wins are the stored losses.
    const std::uint32_t* table = runoutOutcomes_.data() + (player == 0 ? runoutOutcomes_.size() / 2 : 0) + row * pairs;
    const unsigned winShift = player == 0 ? 0 : 16, lossShift = 16 - winShift;
    std::array<float, kMaxHands> wins{}, losses{};
    for (std::size_t other = 0; other < opponentCount; ++other)
    {
        const float reach = opponentReach[other];
        const std::uint32_t* entries = table + other * count;
        for (std::size_t hand = 0; hand < count; ++hand)
        {
            wins[hand] += reach * static_cast<float>(static_cast<int>((entries[hand] >> winShift) & 0xffffu));
            losses[hand] += reach * static_cast<float>(static_cast<int>((entries[hand] >> lossShift) & 0xffffu));
        }
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
        // Every runout row in both layouts: the flop row and one row per possible turn
        // card, an upper bound that spares walking the tree here.
        prepareTraining ? 2 * 53 * handCounts[0] * handCounts[1] * sizeof(std::uint32_t) : 0,
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

void HandTraversal::MatchRegrets(const Node& node, const TrainState& train, float* current, const float* actorReach) const
{
    const auto count = hands[node.actor].size();
    const auto actions = node.childCount;
    const std::int16_t* regrets = train.regrets + node.strategyOffset;
    const float uniform = 1.0f / actions;
    // Eight-hand chunks without reach skip their regret loads, most of them once play prunes
    // lines; the zero policy leaves the propagated reach exactly as any policy would.
    const std::size_t aligned = actorReach ? count / 8 * 8 : 0;
    for (std::size_t chunk = 0; chunk < aligned; chunk += 8)
    {
#if defined(__AVX2__)
        const __m256 reach = _mm256_loadu_ps(actorReach + chunk);
        if (_mm256_movemask_ps(_mm256_cmp_ps(reach, _mm256_setzero_ps(), _CMP_NEQ_UQ)) == 0)
        {
            for (std::size_t action = 0; action < actions; ++action)
                _mm256_storeu_ps(current + action * count + chunk, _mm256_setzero_ps());
            continue;
        }
        __m256 positive = _mm256_setzero_ps();
        for (std::size_t action = 0; action < actions; ++action)
        {
            const __m256 clipped = _mm256_max_ps(vector::Load(regrets + action * count + chunk), _mm256_setzero_ps());
            _mm256_storeu_ps(current + action * count + chunk, clipped);
            positive = _mm256_add_ps(positive, clipped);
        }
        const __m256 matched = _mm256_cmp_ps(positive, _mm256_setzero_ps(), _CMP_GT_OQ);
        for (std::size_t action = 0; action < actions; ++action)
        {
            float* row = current + action * count + chunk;
            _mm256_storeu_ps(row, _mm256_blendv_ps(_mm256_set1_ps(uniform), _mm256_div_ps(_mm256_loadu_ps(row), positive), matched));
        }
#else
        bool any = false;
        for (int i = 0; i < 8; ++i)
            any |= actorReach[chunk + i] != 0.0f;
        if (any)
            MatchHands(regrets, current, actions, count, uniform, chunk, chunk + 8);
        else
            for (std::size_t action = 0; action < actions; ++action)
                std::fill_n(current + action * count + chunk, 8, 0.0f);
#endif
    }
    MatchHands(regrets, current, actions, count, uniform, aligned, count);
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
                MatchRegrets(node, train, policy, reach);
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
    const std::uint16_t* strategySums,
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
        else if (train && !runoutOutcomes_.empty())
            EvaluateRunoutOutcomes(node, player, opponentReach, divisors, values);
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
            MatchRegrets(node, *train, current, node.actor == player ? nullptr : opponentReach);
        }
        else if (strategySums)
        {
            const std::uint16_t* sums = strategySums + node.strategyOffset;
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
    const auto descend = [&](std::size_t action, float* output)
    {
        const auto childIndex = children[node.childOffset + action];
        const float* childReach =
            PropagateChild(nodeIndex, action, 1 - player, nodeStrategy, opponentReach, workspace.reach.data() + (depth + 1) * stride);
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
    // The opponent's decisions accumulate its cumulative strategy from the reach they
    // propagate, reusing the finished child value rows as scratch.
    if (train && node.kind == Kind::Decision && !acting)
        AccumulateNodeStrategy(
            train->strategySums + node.strategyOffset,
            node.strategyOffset,
            node.childCount,
            opponentCount,
            opponentReach,
            nodeStrategy,
            childrenValues,
            train->weights.averageWeight,
            train->weights.update
        );
    if (acting && train && live)
    {
        const auto& weights = train->weights;
        const std::uint32_t skipped = weights.update - 1 - train->stamps[nodeIndex];
        const float halving = std::ldexp(1.0f, -static_cast<int>(std::min<std::uint32_t>(skipped, 200)));
        UpdateNodeRegrets(
            train->regrets + node.strategyOffset,
            node.strategyOffset,
            node.childCount,
            count,
            childrenValues,
            values,
            weights.positiveScale,
            weights.positiveInverse,
            halving,
            weights.update
        );
        train->stamps[nodeIndex] = weights.update;
    }
    return live;
}
} // namespace solver::engine
