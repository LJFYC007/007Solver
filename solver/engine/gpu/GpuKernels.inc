// Shared FP32 arithmetic, compiled as CUDA C++ or Metal Shading Language.
using namespace solver::engine::gpu;
#ifdef __METAL_VERSION__
#define GPU_INLINE inline
#define GPU_DEVICE device
#define GPU_BUFFER(type, name, index) device type* name [[buffer(index)]]
#define GPU_ENTRY(name) kernel void name
#define GPU_TAIL                                                                                                                     \
    constant State &state [[buffer(StateBuffer)]], constant Pass &pass [[buffer(PassBuffer)]], uint gid [[thread_position_in_grid]], \
        uint simdWidth [[threads_per_simdgroup]]
#define GPU_INDEX gid
#define GPU_ITEM_INDEX (gid / state.stride)
#define GPU_HAND_INDEX (gid % state.stride)
#define GPU_WIDTH simdWidth
#define GPU_SHARED threadgroup
#define GPU_GROUP_TAIL                                                                                    \
    , threadgroup float *groupMemory [[threadgroup(0)]], uint localIndex [[thread_index_in_threadgroup]], \
        uint groupIndex [[threadgroup_position_in_grid]], uint groupSize [[threads_per_threadgroup]]
#define GPU_GROUP_MEMORY
#define GPU_GROUP_INDEX groupIndex
#define GPU_LOCAL_INDEX localIndex
#define GPU_GROUP_SIZE groupSize
#define GPU_GROUP_BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)
#define GPU_LDEXP(value, exponent) ldexp(value, exponent)
#else
#define GPU_INLINE __device__ __forceinline__
#define GPU_DEVICE
#define GPU_BUFFER(type, name, index) type* name
#define GPU_ENTRY(name) __global__ void name
#define GPU_TAIL const State *statePointer, Pass pass
#define GPU_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define GPU_ITEM_INDEX blockIdx.x
#define GPU_HAND_INDEX (blockIdx.y * blockDim.x + threadIdx.x)
#define GPU_WIDTH 32u
#define GPU_SHARED
#define GPU_GROUP_TAIL
#define GPU_GROUP_MEMORY extern __shared__ float groupMemory[];
#define GPU_GROUP_INDEX blockIdx.x
#define GPU_LOCAL_INDEX threadIdx.x
#define GPU_GROUP_SIZE blockDim.x
#define GPU_GROUP_BARRIER() __syncthreads()
#define GPU_LDEXP(value, exponent) ldexpf(value, exponent)
#endif
#define GPU_ARGUMENTS                                                                                                                    \
    GPU_BUFFER(const Node, nodes, NodesBuffer), GPU_BUFFER(const U32, childSlots, ChildSlotsBuffer),                                     \
        GPU_BUFFER(const Hand, hands, HandsBuffer), GPU_BUFFER(const unsigned short, ranks, RanksBuffer),                                \
        GPU_BUFFER(const int, runouts, RunoutsBuffer), GPU_BUFFER(const U32, order, OrderBuffer),                                        \
        GPU_BUFFER(const U32, cards, CardsBuffer), GPU_BUFFER(U32, outcomes, OutcomesBuffer), GPU_BUFFER(float, regrets, RegretsBuffer), \
        GPU_BUFFER(float, sums, SumsBuffer), GPU_BUFFER(float, scratch, ScratchBuffer), GPU_BUFFER(float, values, ValuesBuffer),         \
        GPU_BUFFER(U32, flags, FlagsBuffer), GPU_BUFFER(U32, stamps, StampsBuffer), GPU_TAIL
#ifdef __METAL_VERSION__
#define GPU_STATE
#else
#define GPU_STATE const State state = *statePointer;
#endif

// Regret matching: positive is the hand's positive-regret sum over all actions.
GPU_INLINE float MatchedProbability(float regret, float positive, U32 actions)
{
    return positive > 0.0f ? (regret > 0.0f ? regret : 0.0f) / positive : 1.0f / float(actions);
}

// One parent normalizes its entry policy once for all children and, while training,
// accumulates its cumulative strategy from the reach it propagates. A thread carries
// two adjacent hands so twice the loads are in flight per warp; pair and value1 are
// zero when only hand h exists. A zero ActionCount keeps the general path; small
// decisions retain entry data.
template<U32 ActionCount>
GPU_INLINE void ReachNode(
    Node n,
    State state,
    U32 p,
    U32 h,
    U32 pair,
    float value0,
    float value1,
    GPU_DEVICE const U32* childSlots,
    GPU_DEVICE const float* regrets,
    GPU_DEVICE float* sums,
    GPU_DEVICE float* scratch
)
{
    const U32 count = p ? state.hands[1] : state.hands[0];
    const U32 actions = ActionCount ? ActionCount : n.count;
    GPU_DEVICE const float* policy = state.evaluation ? sums : regrets;
    const bool accumulate = !state.evaluation;
    U32 slots[ActionCount ? ActionCount : 1];
    float entries0[ActionCount ? ActionCount : 1], entries1[ActionCount ? ActionCount : 1];
    float sums0[ActionCount ? ActionCount : 1], sums1[ActionCount ? ActionCount : 1];
    if (ActionCount)
        for (U32 a = 0; a < actions; ++a)
            slots[a] = n.childSlot[a];
    // Zero reach still clears every child slot; only its policy and sum loads are skipped.
    float positive0 = 0.0f, positive1 = 0.0f;
    for (U32 a = 0; a < actions; ++a)
    {
        const U64 entry = n.strategy + U64(a) * count + h;
        const float x0 = value0 != 0.0f ? policy[entry] : 0.0f;
        const float x1 = value1 != 0.0f ? policy[entry + 1] : 0.0f;
        if (ActionCount)
        {
            entries0[a] = x0;
            entries1[a] = x1;
            sums0[a] = accumulate && value0 != 0.0f ? sums[entry] : 0.0f;
            sums1[a] = accumulate && value1 != 0.0f ? sums[entry + 1] : 0.0f;
        }
        positive0 += x0 > 0.0f ? x0 : 0.0f;
        positive1 += x1 > 0.0f ? x1 : 0.0f;
    }
    for (U32 a = 0; a < actions; ++a)
    {
        const U64 entry = n.strategy + U64(a) * count + h;
        float child0 = value0, child1 = value1;
        if (value0 != 0.0f)
            child0 *= MatchedProbability(ActionCount ? entries0[a] : policy[entry], positive0, actions);
        if (value1 != 0.0f)
            child1 *= MatchedProbability(ActionCount ? entries1[a] : policy[entry + 1], positive1, actions);
        GPU_DEVICE float* target = scratch + U64(ActionCount ? slots[a] : childSlots[n.edge + a]) * state.stride + h;
        target[0] = child0;
        if (pair)
            target[1] = child1;
        // Zero increments leave the cumulative strategy untouched.
        if (accumulate)
        {
            if (child0 != 0.0f)
                sums[entry] = (ActionCount ? sums0[a] : sums[entry]) + state.averageWeight * child0;
            if (child1 != 0.0f)
                sums[entry + 1] = (ActionCount ? sums1[a] : sums[entry + 1]) + state.averageWeight * child1;
        }
    }
}

GPU_ENTRY(Reach)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U32 index = GPU_INDEX, item = index / pass.lanes, lane = index % pass.lanes;
    if (item >= pass.count)
        return;
    const Node n = nodes[pass.offset + item];
    // Only the opponent's reach is propagated, one row of stride floats per slot: terminals
    // read it and the opponent's cumulative strategy accumulates it. The updating player's
    // own reach is never needed.
    const U32 p = 1 - state.player, count = p ? state.hands[1] : state.hands[0], base = p ? state.hands[0] : 0;
    U32 h = lane, pair = 0;
    float value0, value1 = 0.0f;
    if (pass.boundary)
    {
        // Region roots derive the opponent's reach from the game root or the dealt card.
        if (h >= count)
            return;
        if (n.parent == kNoIndex)
            value0 = hands[base + h].weight;
        else
        {
            const Node parent = nodes[n.parent];
            value0 = scratch[U64(p ? parent.reachSlot[1] : parent.reachSlot[0]) * state.stride + h];
            value0 = (hands[base + h].mask & n.board & ~parent.board) ? 0.0f : value0 * (1.0f / float(parent.count - 4));
        }
        scratch[U64(n.slot) * state.stride + h] = value0;
        if (n.kind != NodeKind::Decision || n.actor != p)
            return;
    }
    else
    {
        // The opponent's deeper decisions propagate its reach, two adjacent hands per thread.
        h = 2 * lane;
        if (h >= count)
            return;
        pair = h + 1 < count ? 1u : 0u;
        GPU_DEVICE const float* source = scratch + U64(p ? n.reachSlot[1] : n.reachSlot[0]) * state.stride + h;
        value0 = source[0];
        value1 = pair ? source[1] : 0.0f;
    }
    if (n.count == 2)
        ReachNode<2>(n, state, p, h, pair, value0, value1, childSlots, regrets, sums, scratch);
    else if (n.count == 3)
        ReachNode<3>(n, state, p, h, pair, value0, value1, childSlots, regrets, sums, scratch);
    else
        ReachNode<0>(n, state, p, h, pair, value0, value1, childSlots, regrets, sums, scratch);
}

// True in every lane of the group when any lane holds true. Metal reduces per SIMD
// group through the leading group memory floats; CUDA has a group-wide primitive.
GPU_INLINE bool GroupAny(bool value, GPU_SHARED float* groupFlags, U32 local, U32 groupWidth, U32 width)
{
#ifdef __METAL_VERSION__
    // All SIMD lanes must participate in the vote before selecting its writer.
    const bool simdAny = simd_any(value);
    if (local % width == 0)
        groupFlags[local / width] = simdAny ? 1.0f : 0.0f;
    GPU_GROUP_BARRIER();
    bool any = false;
    for (U32 g = 0; g < (groupWidth + width - 1) / width; ++g)
        any |= groupFlags[g] != 0.0f;
    return any;
#else
    (void)groupFlags;
    (void)local;
    (void)groupWidth;
    (void)width;
    return __syncthreads_or(value);
#endif
}

// All lanes of a complete SIMD group participate, including zero-padded tails.
GPU_INLINE float GroupScan(float value, U32 lane, U32 width)
{
    for (U32 offset = 1; offset < width; offset *= 2)
    {
#ifdef __METAL_VERSION__
        const float previous = simd_shuffle_up(value, offset);
#else
        const float previous = __shfl_up_sync(0xffffffffu, value, offset);
#endif
        if (lane >= offset)
            value += previous;
    }
    return value;
}

GPU_INLINE float GroupLast(float value, U32 width)
{
#ifdef __METAL_VERSION__
    return simd_shuffle(value, width - 1);
#else
    return __shfl_sync(0xffffffffu, value, width - 1);
#endif
}

// Separate SIMD groups scan forward/reverse reach concurrently. Independent
// sums avoid subtracting nearly equal cumulative totals for loss mass.
GPU_INLINE void TerminalPrefix(
    Node n,
    State state,
    U32 lane,
    U32 width,
    bool reverse,
    GPU_DEVICE const U32* order,
    GPU_SHARED const float* reach,
    GPU_SHARED float* prefix
)
{
    if (n.kind == NodeKind::ForcedRunout)
        return;
    const U32 p = 1 - state.player, count = p ? state.hands[1] : state.hands[0];
    const U32 rankCount = p ? n.rankCounts[1] : n.rankCounts[0];
    const U32 total = state.hands[0] + state.hands[1], base = p ? state.hands[0] : 0;
    float sum = 0.0f;
    if (n.kind == NodeKind::Fold)
    {
        if (reverse)
            return;
        for (U32 h = lane; h < count; h += width)
            sum += reach[h];
        sum = GroupScan(sum, lane, width);
        if (lane == width - 1)
            prefix[0] = sum;
        return;
    }
    GPU_DEVICE const U32* sorted = order + U64(n.rankRow) * 3 * total + base;
    for (U32 begin = 0; begin < rankCount; begin += width)
    {
        const U32 offset = begin + lane, i = reverse ? rankCount - 1 - offset : offset;
        const float value = offset < rankCount ? reach[sorted[i]] : 0.0f;
        const float partial = sum + GroupScan(value, lane, width);
        if (offset < rankCount)
            prefix[(reverse ? state.stride : 0) + i] = partial;
        sum = GroupLast(partial, width);
    }
}

GPU_ENTRY(Terminal)(GPU_ARGUMENTS GPU_GROUP_TAIL)
{
    GPU_STATE
    GPU_GROUP_MEMORY
    const U32 item = GPU_GROUP_INDEX, local = GPU_LOCAL_INDEX, groupWidth = GPU_GROUP_SIZE;
    const U32 p = state.player, opponent = 1 - p;
    // Fixed indices keep the small State/Node arrays in registers on CUDA.
    const U32 count = p ? state.hands[1] : state.hands[0], opponentCount = p ? state.hands[0] : state.hands[1];
    if (item >= pass.count)
        return;
    const Node n = nodes[pass.offset + item];
    const U32 total = state.hands[0] + state.hands[1], mineBase = p ? state.hands[0] : 0;
    const U32 reachSlot = p ? n.reachSlot[0] : n.reachSlot[1];
    GPU_SHARED float* reach = groupMemory + kGroupFlags;
    GPU_SHARED float* prefix = reach + state.stride;
    GPU_SHARED float* blockers = prefix + (2 * state.stride > 53u ? 2 * state.stride : 53u);
    bool nonzero = false;
    for (U32 other = local; other < opponentCount; other += groupWidth)
    {
        const float r = scratch[U64(reachSlot) * state.stride + other];
        reach[other] = r;
        nonzero |= r != 0.0f;
    }
    // A terminal without opponent reach has zero values, which Backup substitutes from
    // the flag instead of reading them.
    const bool live = GroupAny(nonzero, groupMemory, local, groupWidth, GPU_WIDTH);
    if (local == 0)
        flags[n.slot] = live ? 1u : 0u;
    if (!live)
        return;
    for (U32 direction = local / GPU_WIDTH; direction < 2; direction += groupWidth / GPU_WIDTH)
        TerminalPrefix(n, state, local % GPU_WIDTH, GPU_WIDTH, direction != 0, order, reach, prefix);
    if (n.kind == NodeKind::Fold)
        for (U32 card = local; card < 52; card += groupWidth)
        {
            float mass = 0.0f;
            for (U32 entry = cards[opponent * kCardListStride + card]; entry < cards[opponent * kCardListStride + card + 1]; ++entry)
                mass += reach[cards[entry]];
            prefix[1 + card] = mass;
        }
    // Showdowns accumulate each card's blocked reach in rank order once per node, as the
    // CPU evaluator's per-card masses do, instead of rescanning blockers per hand.
    if (n.kind == NodeKind::Showdown)
        for (U32 card = local; card < 52; card += groupWidth)
        {
            const U32 begin = cards[opponent * kCardListStride + card], end = cards[opponent * kCardListStride + card + 1];
            GPU_DEVICE const unsigned short* sorted = ranks + U64(n.rankRow) * 3 * total + total + (begin - kCardListHeader);
            GPU_SHARED float* run = blockers + (begin - cards[opponent * kCardListStride]) + card;
            float sum = 0.0f;
            run[0] = 0.0f;
            for (U32 k = 0; k < end - begin; ++k)
            {
                sum += reach[sorted[k]];
                run[k + 1] = sum;
            }
        }
    GPU_GROUP_BARRIER();
    GPU_DEVICE float* output = values + U64(n.slot) * state.stride;
    for (U32 h = local; h < count; h += groupWidth)
    {
        const Hand mine = hands[mineBase + h];
        if ((mine.mask & n.board) || mine.divisor <= 0.0f)
        {
            output[h] = 0.0f;
            continue;
        }
        if (n.kind == NodeKind::Fold)
        {
            float mass = prefix[0] - prefix[1 + mine.card0] - prefix[1 + mine.card1];
            if (mine.matching >= 0)
                mass += reach[mine.matching];
            output[h] = (mass / mine.divisor) * (p == 0 ? n.utility[0] : -n.utility[0]);
            continue;
        }
        const float win = p == 0 ? n.utility[0] : -n.utility[2];
        const float tie = p == 0 ? n.utility[1] : -n.utility[1];
        const float loss = p == 0 ? n.utility[2] : -n.utility[0];
        float mass = 0.0f, wins = 0.0f, losses = 0.0f;
        if (n.kind == NodeKind::ForcedRunout)
        {
            for (U32 other = 0; other < opponentCount; ++other)
            {
                const U64 pair = p == 0 ? U64(h) * state.hands[1] + other : U64(other) * state.hands[1] + h;
                const U32 counts = outcomes[U64(n.outcomeRow) * state.hands[0] * state.hands[1] + pair];
                // Overlapping pairs hold the sentinel and contribute nothing; no branch per pair.
                const float r = counts == 0xffffffffu ? 0.0f : reach[other];
                mass += r;
                wins += r * float(p == 0 ? counts & 0xffffu : counts >> 16);
                losses += r * float(p == 0 ? counts >> 16 : counts & 0xffffu);
            }
            const float divisor = n.outcomeRow == 0 ? 990.0f : 44.0f;
            output[h] = (tie * mass + (win - tie) * (wins / divisor) + (loss - tie) * (losses / divisor)) / mine.divisor;
            continue;
        }
        const U32 legal = p ? n.rankCounts[0] : n.rankCounts[1];
        // Rank boundaries and blocker positions depend only on the board and ranges.
        GPU_DEVICE const U32* row = order + U64(n.rankRow) * 3 * total;
        const U32 bounds = row[total + mineBase + h], positions = row[2 * total + mineBase + h];
        const U32 lower = bounds & 0xffffu, upper = bounds >> 16;
        wins = lower ? prefix[lower - 1] : 0.0f;
        losses = upper < legal ? prefix[state.stride + upper] : 0.0f;
        // As in the CPU evaluator, zero-tie showdowns need only win/loss masses.
        const bool needMass = tie != 0.0f;
        mass = needMass && legal ? prefix[legal - 1] : 0.0f;
        float blocked = 0.0f, blockedWins = 0.0f, blockedLosses = 0.0f;
        for (U32 c = 0; c < 2; ++c)
        {
            const U32 card = c == 0 ? mine.card0 : mine.card1;
            const U32 begin = cards[opponent * kCardListStride + card], length = cards[opponent * kCardListStride + card + 1] - begin;
            GPU_SHARED const float* run = blockers + (begin - cards[opponent * kCardListStride]) + card;
            const U32 below = (positions >> (16 * c)) & 0xffu, through = (positions >> (16 * c + 8)) & 0xffu;
            blocked += run[length];
            blockedWins += run[below];
            // A card's list has at most 51 entries, so this difference loses only ulps of that card's mass.
            blockedLosses += run[length] - run[through];
        }
        if (needMass)
        {
            mass -= blocked;
            if (mine.matching >= 0)
                mass += reach[mine.matching];
        }
        wins -= blockedWins;
        losses -= blockedLosses;
        output[h] = (tie * mass + (win - tie) * wins + (loss - tie) * losses) / mine.divisor;
    }
}

// A zero ActionCount keeps the general path; small decisions retain entry data
// and issue every load before the first store so latencies overlap. A node is live
// when any child subtree carries opponent reach; pruned nodes have zero values and
// keep their regrets, whose skipped discounts the stamps apply at the next live update.
template<U32 ActionCount>
GPU_INLINE void BackupNode(
    Node n,
    State state,
    U32 h,
    GPU_DEVICE const U32* childSlots,
    GPU_DEVICE float* regrets,
    GPU_DEVICE float* values,
    GPU_DEVICE U32* flags,
    GPU_DEVICE U32* stamps
)
{
    const U32 count = state.player ? state.hands[1] : state.hands[0];
    const U32 actions = ActionCount ? ActionCount : n.count;
    const bool acting = n.kind == NodeKind::Decision && n.actor == state.player;
    const bool training = acting && !state.evaluation;
    U32 slots[ActionCount ? ActionCount : 1];
    U32 lives[ActionCount ? ActionCount : 1];
    float entryRegrets[ActionCount ? ActionCount : 1];
    float childValues[ActionCount ? ActionCount : 1];
    U32 live = 0;
    for (U32 a = 0; a < actions; ++a)
    {
        const U32 slot = ActionCount ? n.childSlot[a] : childSlots[n.edge + a];
        const U32 flag = flags[slot];
        if (ActionCount)
        {
            slots[a] = slot;
            lives[a] = flag;
        }
        live |= flag;
    }
    if (h == 0)
        flags[n.slot] = live;
    float halving = 1.0f;
    if (training)
    {
        // Stamps alternate between two halves: a pass reads the half its player's previous
        // update wrote while writing the other, and every acting decision writes its entry.
        const U32 read = (state.update & 1u) * state.stampCount, write = ((state.update + 1u) & 1u) * state.stampCount;
        const U32 stamp = stamps[read + n.stamp];
        if (h == 0)
            stamps[write + n.stamp] = live ? state.update : stamp;
        const U32 skipped = state.update - 1u - stamp;
        halving = GPU_LDEXP(1.0f, -int(skipped < 200u ? skipped : 200u));
    }
    if (!live)
    {
        if (n.parent == kNoIndex)
            values[U64(n.slot) * state.stride + h] = 0.0f;
        return;
    }
    // Reuse the entry-policy denominator for value backup and the regret update.
    float positive = 0.0f;
    if (training)
        for (U32 a = 0; a < actions; ++a)
        {
            const float r = regrets[n.strategy + U64(a) * count + h];
            if (ActionCount)
                entryRegrets[a] = r;
            positive += r > 0.0f ? r : 0.0f;
        }
    float value = 0.0f;
    for (U32 a = 0; a < actions; ++a)
    {
        const U32 slot = ActionCount ? slots[a] : childSlots[n.edge + a];
        const float child = (ActionCount ? lives[a] : flags[slot]) ? values[U64(slot) * state.stride + h] : 0.0f;
        if (ActionCount)
            childValues[a] = child;
        if (acting && state.evaluation)
            value = a == 0 || child > value ? child : value;
        else if (acting)
        {
            const float r = ActionCount ? entryRegrets[a] : regrets[n.strategy + U64(a) * count + h];
            value += child * MatchedProbability(r, positive, actions);
        }
        else
            value += child;
    }
    values[U64(n.slot) * state.stride + h] = value;
    if (training)
        for (U32 a = 0; a < actions; ++a)
        {
            const U64 entry = n.strategy + U64(a) * count + h;
            const float old = ActionCount ? entryRegrets[a] : regrets[entry];
            const U32 slot = ActionCount ? slots[a] : childSlots[n.edge + a];
            const float child = ActionCount ? childValues[a] : (flags[slot] ? values[U64(slot) * state.stride + h] : 0.0f);
            // Positive regrets are stored divided by positiveScale, so the discounts of the
            // updates this node skipped are already in place; negative regrets halve once
            // per skipped update.
            const float regret = (old > 0.0f ? old * state.positiveScale : old * halving) + (child - value);
            regrets[entry] = regret > 0.0f ? regret * state.positiveInverse : regret * 0.5f;
        }
}

GPU_ENTRY(Backup)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U32 item = GPU_ITEM_INDEX, h = GPU_HAND_INDEX;
    const U32 count = state.player ? state.hands[1] : state.hands[0];
    if (item >= pass.count || h >= count)
        return;
    const Node n = nodes[pass.offset + item];
    if (n.count == 2)
        BackupNode<2>(n, state, h, childSlots, regrets, values, flags, stamps);
    else if (n.count == 3)
        BackupNode<3>(n, state, h, childSlots, regrets, values, flags, stamps);
    else
        BackupNode<0>(n, state, h, childSlots, regrets, values, flags, stamps);
}

GPU_ENTRY(Outcomes)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U64 index = GPU_INDEX, pairs = U64(state.hands[0]) * state.hands[1];
    if (index >= pass.count)
        return;
    const U32 row = U32(index / pairs), h0 = U32(index % pairs / state.hands[1]), h1 = U32(index % state.hands[1]);
    const U64 mask0 = hands[h0].mask, mask1 = hands[state.hands[0] + h1].mask;
    const U64 board = state.board;
    U32 wins = 0, losses = 0;
    // Overlapping private cards store a sentinel that Terminal's runout loop tests instead of masks.
    if (mask0 & mask1)
    {
        outcomes[index] = 0xffffffffu;
        return;
    }
    if (pass.outcomeStage == OutcomeStage::SumTurns)
    {
        for (U32 r = 1; r < state.outcomeRows; ++r)
        {
            const U32 result = outcomes[U64(r) * pairs + index];
            wins += result & 0xffffu;
            losses += result >> 16;
        }
        wins /= 2;
        losses /= 2;
    }
    else if (row || state.outcomeRows == 1)
    {
        const U64 excluded = board | mask0 | mask1;
        const U32 firstBegin = row ? row - 1 : 0, firstEnd = row ? row : 52;
        for (U32 first = firstBegin; first < firstEnd; ++first)
        {
            if (excluded & (U64(1) << first))
                continue;
            for (U32 second = 0; second < (row ? 52 : first); ++second)
            {
                if (first == second || (excluded & (U64(1) << second)))
                    continue;
                const U32 high = first > second ? first : second, low = first < second ? first : second;
                // Matches core::CardPairIndex.
                const int rankRow = runouts[high * (high - 1) / 2 + low];
                const U64 offset = U64(rankRow) * 3 * (state.hands[0] + state.hands[1]);
                const U32 a = ranks[offset + h0], b = ranks[offset + state.hands[0] + h1];
                wins += a > b;
                losses += a < b;
            }
        }
    }
    outcomes[index] = wins | (losses << 16);
}
