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
#endif
#define GPU_ARGUMENTS                                                                                                                    \
    GPU_BUFFER(const Node, nodes, NodesBuffer), GPU_BUFFER(const U32, childSlots, ChildSlotsBuffer),                                     \
        GPU_BUFFER(const Hand, hands, HandsBuffer), GPU_BUFFER(const unsigned short, ranks, RanksBuffer),                                \
        GPU_BUFFER(const int, runouts, RunoutsBuffer), GPU_BUFFER(const U32, order, OrderBuffer),                                        \
        GPU_BUFFER(const U32, cards, CardsBuffer), GPU_BUFFER(U32, outcomes, OutcomesBuffer), GPU_BUFFER(float, regrets, RegretsBuffer), \
        GPU_BUFFER(float, sums, SumsBuffer), GPU_BUFFER(float, scratch, ScratchBuffer), GPU_BUFFER(float, values, ValuesBuffer), GPU_TAIL
#ifdef __METAL_VERSION__
#define GPU_STATE
#else
#define GPU_STATE const State state = *statePointer;
#endif

// One parent/hand normalizes its entry policy once for all children. A zero
// ActionCount keeps the general path; small decisions retain entry data.
template<U32 ActionCount>
GPU_INLINE void ReachNode(
    Node n,
    State state,
    U32 p,
    U32 h,
    U32 skip,
    float value,
    GPU_DEVICE const U32* childSlots,
    GPU_DEVICE const float* regrets,
    GPU_DEVICE const float* sums,
    GPU_DEVICE float* scratch
)
{
    const U32 count = p ? state.hands[1] : state.hands[0];
    const U32 total = state.hands[0] + state.hands[1], base = p ? state.hands[0] : 0;
    const U32 actions = ActionCount ? ActionCount : n.count;
    U32 slots[ActionCount ? ActionCount : 1];
    float entries[ActionCount ? ActionCount : 1];
    if (ActionCount)
        for (U32 a = 0; a < actions; ++a)
            slots[a] = n.childSlot[a];
    // Zero reach still clears every child slot; only the policy lookup is skipped.
    float positive = 0.0f;
    if (value != 0.0f)
        for (U32 a = 0; a < actions; ++a)
        {
            const U64 entry = n.strategy + U64(a) * count + h;
            const float x = state.evaluation ? sums[entry] : regrets[entry];
            if (ActionCount)
                entries[a] = x;
            positive += x > 0.0f ? x : 0.0f;
        }
    for (U32 a = 0; a < actions; ++a)
    {
        if (a < 32 && ((skip >> a) & 1u))
            continue;
        float child = value;
        if (value != 0.0f)
        {
            const U64 entry = n.strategy + U64(a) * count + h;
            const float x = ActionCount ? entries[a] : state.evaluation ? sums[entry] : regrets[entry];
            child *= positive > 0.0f ? (x > 0.0f ? x : 0.0f) / positive : 1.0f / float(actions);
        }
        scratch[U64(ActionCount ? slots[a] : childSlots[n.edge + a]) * total + base + h] = child;
    }
}

GPU_ENTRY(Reach)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U32 index = GPU_INDEX, item = index / pass.lanes, lane = index % pass.lanes;
    if (item >= pass.count)
        return;
    const Node n = nodes[pass.offset + item];
    const U32 total = state.hands[0] + state.hands[1];
    U32 p = n.actor, h = lane;
    float value;
    if (pass.boundary)
    {
        // Region roots derive both players' reach from the game root or the dealt card.
        p = lane < state.hands[0] ? 0 : 1;
        h = p ? lane - state.hands[0] : lane;
        // Own reach is consumed only by training updates.
        if (p == state.player && state.evaluation)
            return;
        const U32 base = p ? state.hands[0] : 0;
        if (n.parent == 0xffffffffu)
            value = p == state.player ? 1.0f : hands[base + h].weight;
        else
        {
            const Node parent = nodes[n.parent];
            value = scratch[U64(p ? parent.reachSlot[1] : parent.reachSlot[0]) * total + base + h];
            value = (hands[base + h].mask & n.board & ~parent.board) ? 0.0f
                                                                     : value * (p == state.player ? 1.0f : 1.0f / float(parent.count - 4));
        }
        scratch[U64(n.slot) * total + base + h] = value;
        if (n.kind != NodeKind::Decision || p != n.actor)
            return;
    }
    else
    {
        if (h >= (p ? state.hands[1] : state.hands[0]) || (p == state.player && state.evaluation))
            return;
        value = scratch[U64(p ? n.reachSlot[1] : n.reachSlot[0]) * total + (p ? state.hands[0] : 0) + h];
    }
    // Terminals read only the opponent's reach, so the updating player's own reach skips them.
    const U32 skip = p == state.player ? n.terminalChildren : 0u;
    if (n.count == 2)
        ReachNode<2>(n, state, p, h, skip, value, childSlots, regrets, sums, scratch);
    else if (n.count == 3)
        ReachNode<3>(n, state, p, h, skip, value, childSlots, regrets, sums, scratch);
    else
        ReachNode<0>(n, state, p, h, skip, value, childSlots, regrets, sums, scratch);
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
    const U32 total = state.hands[0] + state.hands[1], mineBase = p ? state.hands[0] : 0, oppBase = p ? 0 : state.hands[0];
    const U32 reachSlot = p ? n.reachSlot[0] : n.reachSlot[1];
    GPU_SHARED float* reach = groupMemory;
    GPU_SHARED float* prefix = groupMemory + state.stride;
    GPU_SHARED float* blockers = prefix + (2 * state.stride > 53u ? 2 * state.stride : 53u);
#ifdef __METAL_VERSION__
    for (U32 other = local; other < opponentCount; other += groupWidth)
        reach[other] = scratch[U64(reachSlot) * total + oppBase + other];
    GPU_GROUP_BARRIER();
#else
    bool nonzero = false;
    for (U32 other = local; other < opponentCount; other += groupWidth)
    {
        const float r = scratch[U64(reachSlot) * total + oppBase + other];
        reach[other] = r;
        nonzero |= r != 0.0f;
    }
    // Every lane participates; zero values still flow through normal Backup.
    if (!__syncthreads_or(nonzero))
    {
        for (U32 h = local; h < count; h += groupWidth)
            values[U64(n.slot) * state.stride + h] = 0.0f;
        return;
    }
#endif
    for (U32 direction = local / GPU_WIDTH; direction < 2; direction += groupWidth / GPU_WIDTH)
        TerminalPrefix(n, state, local % GPU_WIDTH, GPU_WIDTH, direction != 0, order, reach, prefix);
    if (n.kind == NodeKind::Fold)
        for (U32 card = local; card < 52; card += groupWidth)
        {
            float mass = 0.0f;
            for (U32 entry = cards[opponent * 53 + card]; entry < cards[opponent * 53 + card + 1]; ++entry)
                mass += reach[cards[entry]];
            prefix[1 + card] = mass;
        }
    // Showdowns accumulate each card's blocked reach in rank order once per node, as the
    // CPU evaluator's per-card masses do, instead of rescanning blockers per hand.
    if (n.kind == NodeKind::Showdown)
        for (U32 card = local; card < 52; card += groupWidth)
        {
            const U32 begin = cards[opponent * 53 + card], end = cards[opponent * 53 + card + 1];
            GPU_DEVICE const unsigned short* sorted = ranks + U64(n.rankRow) * 3 * total + total + (begin - 106);
            GPU_SHARED float* run = blockers + (begin - cards[opponent * 53]) + card;
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
            const U32 begin = cards[opponent * 53 + card], length = cards[opponent * 53 + card + 1] - begin;
            GPU_SHARED const float* run = blockers + (begin - cards[opponent * 53]) + card;
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
// and issue every load before the first store so latencies overlap.
template<U32 ActionCount>
GPU_INLINE void BackupNode(
    Node n,
    State state,
    U32 h,
    GPU_DEVICE const U32* childSlots,
    GPU_DEVICE float* regrets,
    GPU_DEVICE float* sums,
    GPU_DEVICE const float* scratch,
    GPU_DEVICE float* values
)
{
    const U32 count = state.player ? state.hands[1] : state.hands[0];
    const U32 actions = ActionCount ? ActionCount : n.count;
    const bool acting = n.kind == NodeKind::Decision && n.actor == state.player;
    const bool training = acting && !state.evaluation;
    U32 slots[ActionCount ? ActionCount : 1];
    float entryRegrets[ActionCount ? ActionCount : 1];
    float entrySums[ActionCount ? ActionCount : 1];
    float childValues[ActionCount ? ActionCount : 1];
    float probabilities[ActionCount ? ActionCount : 1];
    if (ActionCount)
        for (U32 a = 0; a < actions; ++a)
            slots[a] = n.childSlot[a];
    // Reuse the entry-policy denominator for value backup and strategy sums.
    float positive = 0.0f, own = 0.0f;
    if (training)
    {
        for (U32 a = 0; a < actions; ++a)
        {
            const U64 entry = n.strategy + U64(a) * count + h;
            const float r = regrets[entry];
            if (ActionCount)
            {
                entryRegrets[a] = r;
                entrySums[a] = sums[entry];
            }
            positive += r > 0.0f ? r : 0.0f;
        }
        const U32 total = state.hands[0] + state.hands[1], base = state.player ? state.hands[0] : 0;
        own = scratch[U64(state.player ? n.reachSlot[1] : n.reachSlot[0]) * total + base + h];
    }
    float value = 0.0f;
    for (U32 a = 0; a < actions; ++a)
    {
        const float child = values[U64(ActionCount ? slots[a] : childSlots[n.edge + a]) * state.stride + h];
        if (ActionCount)
            childValues[a] = child;
        if (acting && state.evaluation)
            value = a == 0 || child > value ? child : value;
        else if (acting)
        {
            const float r = ActionCount ? entryRegrets[a] : regrets[n.strategy + U64(a) * count + h];
            const float probability = positive > 0.0f ? (r > 0.0f ? r : 0.0f) / positive : 1.0f / float(actions);
            if (ActionCount)
                probabilities[a] = probability;
            value += child * probability;
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
            const float probability = ActionCount       ? probabilities[a]
                                      : positive > 0.0f ? (old > 0.0f ? old : 0.0f) / positive
                                                        : 1.0f / float(actions);
            const float child = ActionCount ? childValues[a] : values[U64(childSlots[n.edge + a]) * state.stride + h];
            const float r = old + (child - value);
            regrets[entry] = r * (r > 0.0f ? state.positiveDiscount : 0.5f);
            sums[entry] = state.averageDiscount * ((ActionCount ? entrySums[a] : sums[entry]) + own * probability);
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
        BackupNode<2>(n, state, h, childSlots, regrets, sums, scratch, values);
    else if (n.count == 3)
        BackupNode<3>(n, state, h, childSlots, regrets, sums, scratch, values);
    else
        BackupNode<0>(n, state, h, childSlots, regrets, sums, scratch, values);
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
