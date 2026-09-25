// Shared FP32 arithmetic, compiled as CUDA C++ or Metal Shading Language.
using namespace solver::engine::gpu;
#ifdef __METAL_VERSION__
#define GPU_INLINE inline
#define GPU_DEVICE device
#define GPU_BUFFER(type, name, index) device type* name [[buffer(index)]]
#define GPU_ENTRY(name) kernel void name
#define GPU_ENTRY_BOUNDED(name, threads, blocks) kernel void name
#define GPU_TAIL                                                                                                                     \
    constant State &state [[buffer(StateBuffer)]], constant Pass &pass [[buffer(PassBuffer)]], uint gid [[thread_position_in_grid]], \
        uint simdWidth [[threads_per_simdgroup]]
#define GPU_INDEX gid
#define GPU_ITEM_INDEX (gid / pass.lanes)
#define GPU_HAND_INDEX (gid % pass.lanes)
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
#define GPU_UINT_AS_FLOAT(value) as_type<float>(value)
#else
#define GPU_INLINE __device__ __forceinline__
#define GPU_DEVICE
#define GPU_BUFFER(type, name, index) type* name
#define GPU_ENTRY(name) __global__ void name
// Caps registers so at least blocks of threads-sized blocks fit one SM; Metal has no equivalent.
#define GPU_ENTRY_BOUNDED(name, threads, blocks) __global__ void __launch_bounds__(threads, blocks) name
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
#define GPU_UINT_AS_FLOAT(value) __uint_as_float(value)
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

// Separate SIMD groups scan a showdown's forward/reverse reach concurrently. Independent
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
    const U32 p = 1 - state.player;
    const U32 rankCount = (n.rankCounts >> (16 * p)) & 0xffffu;
    const U32 total = state.hands[0] + state.hands[1], base = p ? state.hands[0] : 0;
    float sum = 0.0f;
    GPU_DEVICE const U32* sorted = order + U64(n.row) * state.orderPitch + 2 * total + base;
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

// Stages the opponent's reach row of slot row after the group flags and returns whether
// it carries reach. A terminal without opponent reach has zero values, which Backup
// substitutes from the flag instead of reading them.
GPU_INLINE bool StageReach(
    State state,
    U32 row,
    U32 local,
    U32 groupWidth,
    U32 width,
    GPU_DEVICE const float* scratch,
    GPU_SHARED float* groupMemory
)
{
    const U32 opponentCount = state.player ? state.hands[0] : state.hands[1];
    GPU_SHARED float* reach = groupMemory + kGroupFlags;
    bool nonzero = false;
    for (U32 other = local; other < opponentCount; other += groupWidth)
    {
        const float r = scratch[U64(row) * state.stride + other];
        reach[other] = r;
        nonzero |= r != 0.0f;
    }
    return GroupAny(nonzero, groupMemory, local, groupWidth, width);
}

// A fold's values from the staged opponent reach: the total less each held card's mass,
// as the CPU evaluator's per-card masses; the totals occupy prefix.
GPU_INLINE void TerminalFold(
    State state,
    U64 board,
    U32 slot,
    float utility,
    U32 local,
    U32 groupWidth,
    U32 width,
    GPU_DEVICE const Hand* hands,
    GPU_DEVICE const U32* cards,
    GPU_DEVICE float* values,
    GPU_SHARED const float* reach,
    GPU_SHARED float* prefix
)
{
    const U32 p = state.player, opponent = 1 - p;
    const U32 count = p ? state.hands[1] : state.hands[0], opponentCount = p ? state.hands[0] : state.hands[1];
    const U32 mineBase = p ? state.hands[0] : 0;
    if (local < width)
    {
        float sum = 0.0f;
        for (U32 h = local; h < opponentCount; h += width)
            sum += reach[h];
        sum = GroupScan(sum, local, width);
        if (local == width - 1)
            prefix[0] = sum;
    }
    for (U32 card = local; card < 52; card += groupWidth)
    {
        float mass = 0.0f;
        for (U32 entry = cards[opponent * kCardListStride + card]; entry < cards[opponent * kCardListStride + card + 1]; ++entry)
            mass += reach[cards[entry]];
        prefix[1 + card] = mass;
    }
    GPU_GROUP_BARRIER();
    GPU_DEVICE float* output = values + U64(slot) * state.stride;
    const float payoff = p == 0 ? utility : -utility;
    for (U32 h = local; h < count; h += groupWidth)
    {
        // The hand's first half: mask, divisor and packed cards (see Hand).
        const uint4 first = ((GPU_DEVICE const uint4*)(hands + mineBase + h))[0];
        const U64 mask = U64(first.x) | (U64(first.y) << 32);
        const float divisor = GPU_UINT_AS_FLOAT(first.z);
        const U32 card0 = first.w & 0xffu, card1 = (first.w >> 8) & 0xffu, identical = first.w >> 16;
        if ((mask & board) || divisor <= 0.0f)
        {
            output[h] = 0.0f;
            continue;
        }
        float mass = prefix[0] - prefix[1 + card0] - prefix[1 + card1];
        if (identical)
            mass += reach[identical - 1];
        output[h] = (mass / divisor) * payoff;
    }
}

// The stored regret after adding increment. Positive regrets are stored divided by
// positiveScale, so the discounts of the updates a node skipped are already in place;
// negative regrets halve once per skipped update.
GPU_INLINE float UpdatedRegret(float old, float increment, float halving, State state)
{
    const float regret = (old > 0.0f ? old * state.positiveScale : old * halving) + increment;
    return regret > 0.0f ? regret * state.positiveInverse : regret * 0.5f;
}

// Advances an acting decision's stamp at a training update and returns the halving of its
// negative regrets for the updates it skipped. Stamps alternate between two halves: a pass
// reads the half its player's previous update wrote while writing the other, and every
// acting decision writes its entry, from the one writer thread.
GPU_INLINE float AdvanceStamp(State state, U32 index, bool live, bool writer, GPU_DEVICE U32* stamps)
{
    const U32 read = (state.update & 1u) * state.stampCount, write = ((state.update + 1u) & 1u) * state.stampCount;
    const U32 stamp = stamps[read + index];
    if (writer)
        stamps[write + index] = live ? state.update : stamp;
    const U32 skipped = state.update - 1u - stamp;
    return GPU_LDEXP(1.0f, -int(skipped < 200u ? skipped : 200u));
}

// One hand of a decision whose only children are the fold and showdown the calling block
// evaluated: BackupNode's update without child loads. Action foldAction is the fold.
GPU_INLINE void BackupLeafPair(
    Node parent,
    State state,
    U32 h,
    U32 foldAction,
    float fold,
    float showdown,
    float halving,
    GPU_DEVICE float* regrets,
    GPU_DEVICE float* values
)
{
    const U32 count = state.player ? state.hands[1] : state.hands[0];
    const float child0 = foldAction == 0 ? fold : showdown, child1 = foldAction == 0 ? showdown : fold;
    float value;
    if (parent.actor != state.player)
        value = child0 + child1;
    else if (state.evaluation)
        value = child1 > child0 ? child1 : child0;
    else
    {
        GPU_DEVICE float* entry0 = regrets + parent.strategy + h;
        GPU_DEVICE float* entry1 = entry0 + count;
        const float r0 = entry0[0], r1 = entry1[0];
        const float positive = (r0 > 0.0f ? r0 : 0.0f) + (r1 > 0.0f ? r1 : 0.0f);
        value = child0 * MatchedProbability(r0, positive, 2) + child1 * MatchedProbability(r1, positive, 2);
        entry0[0] = UpdatedRegret(r0, child0 - value, halving, state);
        entry1[0] = UpdatedRegret(r1, child1 - value, halving, state);
    }
    values[U64(parent.slot) * state.stride + h] = value;
}

// Registers are capped so the blocks the group memory allows per SM stay resident (21;
// RTX 30 runs at most 16 blocks per SM and ignores a larger hint).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
#define GPU_TERMINAL_BLOCKS 16
#else
#define GPU_TERMINAL_BLOCKS 21
#endif
GPU_ENTRY_BOUNDED(Terminal, 64, GPU_TERMINAL_BLOCKS)(GPU_ARGUMENTS GPU_GROUP_TAIL)
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
    if (n.kind == NodeKind::Fold)
    {
        // A fold without a showdown sibling.
        const bool live = StageReach(state, reachSlot, local, groupWidth, GPU_WIDTH, scratch, groupMemory);
        if (local == 0)
            flags[n.slot] = live ? 1u : 0u;
        if (live)
            TerminalFold(state, n.board, n.slot, n.utility[0], local, groupWidth, GPU_WIDTH, hands, cards, values, reach, prefix);
        return;
    }
    // A showdown evaluates its fold sibling from the same opponent reach row when the
    // updating player folds there, otherwise from the fold's own row first, and backs the
    // parent up when the two are its only children (see Node).
    const bool sibling = n.fold != kNoIndex;
    // Only a showdown with a fold sibling reads its parent's record.
    Node parent{};
    if (sibling)
        parent = nodes[n.parent];
    const bool ownFold = sibling && parent.actor == p;
    const bool backup = sibling && parent.count == 2;
    const U32 foldAction = parent.childSlot[0] == n.fold ? 0u : 1u;
    bool liveFold = false;
    if (sibling && !ownFold)
    {
        // The fold's own row, propagated by Reach; a fused parent reads its values back.
        liveFold = StageReach(state, n.fold, local, groupWidth, GPU_WIDTH, scratch, groupMemory);
        if (liveFold)
            TerminalFold(state, n.board, n.fold, n.foldUtility, local, groupWidth, GPU_WIDTH, hands, cards, values, reach, prefix);
        GPU_GROUP_BARRIER();
    }
    const bool live = StageReach(state, reachSlot, local, groupWidth, GPU_WIDTH, scratch, groupMemory);
    if (ownFold)
        liveFold = live;
    const bool liveParent = live || liveFold;
    // As in BackupNode: an acting parent's stamp advances at every training update.
    const float halving =
        backup && parent.actor == p && !state.evaluation ? AdvanceStamp(state, parent.stamp, liveParent, local == 0, stamps) : 1.0f;
    if (local == 0)
    {
        if (backup)
            flags[parent.slot] = liveParent ? 1u : 0u;
        else
        {
            flags[n.slot] = live ? 1u : 0u;
            if (sibling)
                flags[n.fold] = liveFold ? 1u : 0u;
        }
    }
    if (!(backup ? liveParent : live))
        return;
    if (live && n.kind == NodeKind::Showdown)
    {
        for (U32 direction = local / GPU_WIDTH; direction < 2; direction += groupWidth / GPU_WIDTH)
            TerminalPrefix(n, state, local % GPU_WIDTH, GPU_WIDTH, direction != 0, order, reach, prefix);
        // Showdowns accumulate each card's blocked reach in rank order once per node, as the
        // CPU evaluator's per-card masses do, instead of rescanning blockers per hand.
        for (U32 card = local; card < 52; card += groupWidth)
        {
            const U32 begin = cards[opponent * kCardListStride + card], end = cards[opponent * kCardListStride + card + 1];
            GPU_DEVICE const unsigned short* sorted = ranks + U64(n.row) * 3 * total + total + (begin - kCardListHeader);
            GPU_SHARED float* run = blockers + (begin - cards[opponent * kCardListStride]) + card;
            float sum = 0.0f;
            run[0] = 0.0f;
            for (U32 k = 0; k < end - begin; ++k)
            {
                sum += reach[sorted[k]];
                run[k + 1] = sum;
            }
        }
    }
    GPU_GROUP_BARRIER();
    GPU_DEVICE float* output = values + U64(n.slot) * state.stride;
    GPU_DEVICE float* foldOutput = values + U64(sibling ? n.fold : n.slot) * state.stride;
    const float win = p == 0 ? n.utility[0] : -n.utility[2];
    const float tie = p == 0 ? n.utility[1] : -n.utility[1];
    const float loss = p == 0 ? n.utility[2] : -n.utility[0];
    const float foldPayoff = p == 0 ? n.foldUtility : -n.foldUtility;
    const U32 legal = (n.rankCounts >> (16 * opponent)) & 0xffffu;
    // As in the CPU evaluator, zero-tie showdowns need only win/loss masses; a fold from
    // the same row needs the mass.
    const bool needMass = tie != 0.0f || ownFold;
    const float rankedMass = live && needMass && legal ? prefix[legal - 1] : 0.0f;
    for (U32 h = local; h < count; h += groupWidth)
    {
        // The hand's first half: mask, divisor and packed cards (see Hand).
        GPU_DEVICE const uint4* record = (GPU_DEVICE const uint4*)(hands + mineBase + h);
        const uint4 first = record[0];
        const U64 mask = U64(first.x) | (U64(first.y) << 32);
        const float divisor = GPU_UINT_AS_FLOAT(first.z);
        const U32 identical = first.w >> 16;
        float showdown = 0.0f, fold = 0.0f;
        if (live && !(mask & n.board) && divisor > 0.0f)
        {
            float mass = 0.0f, wins = 0.0f, losses = 0.0f;
            if (n.kind == NodeKind::ForcedRunout)
            {
                // This player's hands are contiguous per opponent hand in its layout of the row
                // (see OutcomesBuffer), so a warp's loads coalesce; eight independent loads per
                // step overlap the row's latency while the sums keep their opponent order.
                const U64 pairs = U64(state.hands[0]) * state.hands[1];
                GPU_DEVICE const U32* table = outcomes + (p == 0 ? U64(state.outcomeRows) * pairs : 0) + U64(n.row) * pairs + h;
#define GPU_RUNOUT_STEP(countsValue, index)                               \
    {                                                                     \
        const U32 counts = (countsValue);                                 \
        /* Overlapping pairs hold the sentinel and contribute nothing. */ \
        const float r = counts == 0xffffffffu ? 0.0f : reach[index];      \
        mass += r;                                                        \
        wins += r * float(p == 0 ? counts & 0xffffu : counts >> 16);      \
        losses += r * float(p == 0 ? counts >> 16 : counts & 0xffffu);    \
    }
                U32 other = 0;
                for (; other + 8 <= opponentCount; other += 8)
                {
                    const U32 c0 = table[U64(other) * count], c1 = table[U64(other + 1) * count];
                    const U32 c2 = table[U64(other + 2) * count], c3 = table[U64(other + 3) * count];
                    const U32 c4 = table[U64(other + 4) * count], c5 = table[U64(other + 5) * count];
                    const U32 c6 = table[U64(other + 6) * count], c7 = table[U64(other + 7) * count];
                    GPU_RUNOUT_STEP(c0, other)
                    GPU_RUNOUT_STEP(c1, other + 1)
                    GPU_RUNOUT_STEP(c2, other + 2)
                    GPU_RUNOUT_STEP(c3, other + 3)
                    GPU_RUNOUT_STEP(c4, other + 4)
                    GPU_RUNOUT_STEP(c5, other + 5)
                    GPU_RUNOUT_STEP(c6, other + 6)
                    GPU_RUNOUT_STEP(c7, other + 7)
                }
                for (; other < opponentCount; ++other)
                    GPU_RUNOUT_STEP(table[U64(other) * count], other)
#undef GPU_RUNOUT_STEP
                const float runouts = n.row == 0 ? 990.0f : 44.0f;
                showdown = (tie * mass + (win - tie) * (wins / runouts) + (loss - tie) * (losses / runouts)) / divisor;
            }
            else
            {
                // Rank boundaries and blocker positions depend only on the board and ranges.
                const uint2 packed = ((GPU_DEVICE const uint2*)(order + U64(n.row) * state.orderPitch))[mineBase + h];
                const U32 lower = packed.x & 0xffffu, upper = packed.x >> 16, positions = packed.y;
                wins = lower ? prefix[lower - 1] : 0.0f;
                losses = upper < legal ? prefix[state.stride + upper] : 0.0f;
                // The hand's second half locates each held card's run over the opponent's holders.
                const uint4 second = record[1];
                float blocked = 0.0f, blockedWins = 0.0f, blockedLosses = 0.0f;
                for (U32 c = 0; c < 2; ++c)
                {
                    const U32 span = c == 0 ? second.y : second.z;
                    GPU_SHARED const float* run = blockers + (span & 0xffffu);
                    const U32 length = span >> 16;
                    const U32 below = (positions >> (16 * c)) & 0xffu, through = (positions >> (16 * c + 8)) & 0xffu;
                    blocked += run[length];
                    blockedWins += run[below];
                    // A card's list has at most 51 entries, so this difference loses only ulps of that card's mass.
                    blockedLosses += run[length] - run[through];
                }
                if (needMass)
                {
                    mass = rankedMass - blocked;
                    if (identical)
                        mass += reach[identical - 1];
                }
                wins -= blockedWins;
                losses -= blockedLosses;
                showdown = (tie * mass + (win - tie) * wins + (loss - tie) * losses) / divisor;
                if (ownFold)
                    fold = (mass / divisor) * foldPayoff;
            }
        }
        if (backup && !ownFold)
            fold = liveFold ? foldOutput[h] : 0.0f;
        if (!backup)
        {
            output[h] = showdown;
            if (ownFold)
                foldOutput[h] = fold;
        }
        else
            BackupLeafPair(parent, state, h, foldAction, fold, showdown, halving, regrets, values);
    }
}

// A zero ActionCount keeps the general path; small decisions retain entry data
// and issue every load before the first store so latencies overlap. A thread carries
// two adjacent hands so twice the loads are in flight per warp; pair is zero when only
// hand h exists. Child values are loaded before their flags are known and replaced by
// zero for dead children, so the flag and value latencies overlap. A node is live when
// any child subtree carries opponent reach; pruned nodes have zero values and keep
// their regrets, whose skipped discounts the stamps apply at the next live update.
template<U32 ActionCount>
GPU_INLINE void BackupNode(
    Node n,
    State state,
    U32 h,
    U32 pair,
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
    float entryRegrets0[ActionCount ? ActionCount : 1], entryRegrets1[ActionCount ? ActionCount : 1];
    float childValues0[ActionCount ? ActionCount : 1], childValues1[ActionCount ? ActionCount : 1];
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
    const float halving = training ? AdvanceStamp(state, n.stamp, live != 0, h == 0, stamps) : 1.0f;
    GPU_DEVICE float* output = values + U64(n.slot) * state.stride + h;
    if (!live)
    {
        if (n.parent == kNoIndex)
        {
            output[0] = 0.0f;
            if (pair)
                output[1] = 0.0f;
        }
        return;
    }
    // Reuse the entry-policy denominator for value backup and the regret update.
    float positive0 = 0.0f, positive1 = 0.0f;
    if (training)
        for (U32 a = 0; a < actions; ++a)
        {
            GPU_DEVICE const float* entry = regrets + n.strategy + U64(a) * count + h;
            const float r0 = entry[0], r1 = pair ? entry[1] : 0.0f;
            if (ActionCount)
            {
                entryRegrets0[a] = r0;
                entryRegrets1[a] = r1;
            }
            positive0 += r0 > 0.0f ? r0 : 0.0f;
            positive1 += r1 > 0.0f ? r1 : 0.0f;
        }
    float value0 = 0.0f, value1 = 0.0f;
    for (U32 a = 0; a < actions; ++a)
    {
        const U32 slot = ActionCount ? slots[a] : childSlots[n.edge + a];
        const U32 flag = ActionCount ? lives[a] : flags[slot];
        GPU_DEVICE const float* source = values + U64(slot) * state.stride + h;
        const float loaded0 = source[0], loaded1 = pair ? source[1] : 0.0f;
        const float child0 = flag ? loaded0 : 0.0f, child1 = flag ? loaded1 : 0.0f;
        if (ActionCount)
        {
            childValues0[a] = child0;
            childValues1[a] = child1;
        }
        if (acting && state.evaluation)
        {
            value0 = a == 0 || child0 > value0 ? child0 : value0;
            value1 = a == 0 || child1 > value1 ? child1 : value1;
        }
        else if (acting)
        {
            GPU_DEVICE const float* entry = regrets + n.strategy + U64(a) * count + h;
            value0 += child0 * MatchedProbability(ActionCount ? entryRegrets0[a] : entry[0], positive0, actions);
            if (pair)
                value1 += child1 * MatchedProbability(ActionCount ? entryRegrets1[a] : entry[1], positive1, actions);
        }
        else
        {
            value0 += child0;
            value1 += child1;
        }
    }
    output[0] = value0;
    if (pair)
        output[1] = value1;
    if (training)
        for (U32 a = 0; a < actions; ++a)
        {
            GPU_DEVICE float* entry = regrets + n.strategy + U64(a) * count + h;
            const U32 slot = ActionCount ? slots[a] : childSlots[n.edge + a];
            GPU_DEVICE const float* source = values + U64(slot) * state.stride + h;
            const float child0 = ActionCount ? childValues0[a] : (flags[slot] ? source[0] : 0.0f);
            entry[0] = UpdatedRegret(ActionCount ? entryRegrets0[a] : entry[0], child0 - value0, halving, state);
            if (pair)
            {
                const float child1 = ActionCount ? childValues1[a] : (flags[slot] ? source[1] : 0.0f);
                entry[1] = UpdatedRegret(ActionCount ? entryRegrets1[a] : entry[1], child1 - value1, halving, state);
            }
        }
}

// A chance node sums its children, whose slots are consecutive from childSlot[0] (see
// Plan), so eight children's flags and values are loaded per round with no dependent
// slot lookup; the sums keep child order. Launches of a few such nodes end an update.
GPU_INLINE void BackupChance(Node n, State state, U32 h, U32 pair, GPU_DEVICE float* values, GPU_DEVICE U32* flags)
{
    U32 live = 0;
    float value0 = 0.0f, value1 = 0.0f;
    for (U32 a = 0; a < n.count; a += 8)
    {
        U32 flag[8];
        float loaded0[8], loaded1[8];
        for (U32 k = 0; k < 8; ++k)
        {
            const bool present = a + k < n.count;
            GPU_DEVICE const float* source = values + U64(n.childSlot[0] + a + k) * state.stride + h;
            flag[k] = present ? flags[n.childSlot[0] + a + k] : 0u;
            loaded0[k] = present ? source[0] : 0.0f;
            loaded1[k] = present && pair ? source[1] : 0.0f;
        }
        for (U32 k = 0; k < 8; ++k)
        {
            live |= flag[k];
            value0 += flag[k] ? loaded0[k] : 0.0f;
            value1 += flag[k] ? loaded1[k] : 0.0f;
        }
    }
    if (h == 0)
        flags[n.slot] = live;
    GPU_DEVICE float* output = values + U64(n.slot) * state.stride + h;
    if (!live && n.parent != kNoIndex)
        return;
    output[0] = value0;
    if (pair)
        output[1] = value1;
}

// 48 registers keep 42 warps per SM resident for both hand-tile sizes instead of 36
// (Reach at 40 registers spills and loses 5%).
GPU_ENTRY_BOUNDED(Backup, 256, 5)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U32 item = GPU_ITEM_INDEX, h = 2 * GPU_HAND_INDEX;
    const U32 count = state.player ? state.hands[1] : state.hands[0];
    if (item >= pass.count || h >= count)
        return;
    const U32 pair = h + 1 < count ? 1u : 0u;
    const Node n = nodes[pass.offset + item];
    if (n.kind == NodeKind::Chance)
        BackupChance(n, state, h, pair, values, flags);
    else if (n.count == 2)
        BackupNode<2>(n, state, h, pair, childSlots, regrets, values, flags, stamps);
    else if (n.count == 3)
        BackupNode<3>(n, state, h, pair, childSlots, regrets, values, flags, stamps);
    else
        BackupNode<0>(n, state, h, pair, childSlots, regrets, values, flags, stamps);
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
    // The player-1-major copy of every row follows the player-0-major rows.
    const U64 transposed = (U64(state.outcomeRows) + row) * pairs + U64(h1) * state.hands[0] + h0;
    U32 wins = 0, losses = 0;
    // Overlapping private cards store a sentinel that Terminal's runout loop tests instead of masks.
    if (mask0 & mask1)
    {
        outcomes[index] = outcomes[transposed] = 0xffffffffu;
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
    outcomes[index] = outcomes[transposed] = wins | (losses << 16);
}
