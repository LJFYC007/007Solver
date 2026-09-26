// CUDA kernels of the FP32 DCFR passes, included once by CudaExecutor.cu.
using namespace solver::engine::gpu;
// Every kernel takes the buffers in BufferIndex order, the last holding the update's State, then
// the pass by value.
#define GPU_ARGUMENTS                                                                                                          \
    const Node *nodes, const Hand *hands, const unsigned short *ranks, const int *runouts, const U32 *order, const U32 *cards, \
        U32 *outcomes, short *regrets, unsigned short *sums, float *scratch, U32 *flags, U32 *stamps, const State *statePointer, Pass pass
// numerator / divisor for regret matching's operands: quantized integers, the divisor at most
// kMaxActions * 65535 < 2^21. For such normal operands CUDA's division fast path, a refined
// reciprocal and a residual correction, is the correctly rounded quotient, and zero numerators
// give +0; `/` would add a range check whose slow-path call every warp with a zero divisor
// (a result discarded below) takes.
__device__ __forceinline__ float RegretRatio(float numerator, float divisor)
{
    float estimate;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(estimate) : "f"(divisor));
    const float reciprocal = __fmaf_rn(estimate, __fmaf_rn(-divisor, estimate, 1.0f), estimate);
    const float quotient = __fmaf_rn(numerator, reciprocal, 0.0f);
    return __fmaf_rn(reciprocal, __fmaf_rn(-divisor, quotient, numerator), quotient);
}

// Regret matching: positive is the hand's positive-regret sum over all actions.
__device__ __forceinline__ float MatchedProbability(float regret, float positive, U32 actions)
{
    return positive > 0.0f ? RegretRatio(regret > 0.0f ? regret : 0.0f, positive) : 1.0f / float(actions);
}

// Hands h and h + 1 of a scratch row (State::stride keeps rows 16-byte aligned, so an even h
// is 8-byte aligned) as one access; a thread without a pair moves hand h alone.
__device__ __forceinline__ float2 LoadPair(const float* row, U32 pair)
{
    return pair ? *(const float2*)row : make_float2(row[0], 0.0f);
}

__device__ __forceinline__ void StorePair(float* row, U32 pair, float value0, float value1)
{
    if (pair)
        *(float2*)row = make_float2(value0, value1);
    else
        row[0] = value0;
}

// A row's float4 group with the lanes past its hands, which hold stale scratch, zeroed; valid
// is the number of hands from the group's first. A group past the row's last hand loads nothing.
__device__ __forceinline__ float4 StagedGroup(const float4* source, U32 group, U32 valid)
{
    const float4 r = valid > 0 ? source[group] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return make_float4(r.x, valid > 1 ? r.y : 0.0f, valid > 2 ? r.z : 0.0f, valid > 3 ? r.w : 0.0f);
}

// Whether a staged group carries reach: a float is nonzero when its bits other than the sign
// are, so the test takes no branches.
__device__ __forceinline__ bool AnyReach(float4 group)
{
    return ((FloatBits(group.x) | FloatBits(group.y) | FloatBits(group.z) | FloatBits(group.w)) << 1) != 0u;
}

// Starts moving the global line holding address into L2 ahead of its load.
__device__ __forceinline__ void PrefetchL2(const void* address)
{
    asm volatile("prefetch.global.L2 [%0];" ::"l"(address));
}

// Quantize (see GpuQuantize.h) of a value that is not negative: one fused multiply-add rounded
// down sums the exact scaled value and the dither, whose floor Quantize returns (a scaled value's
// own rounding only matters below 2^-126, where both floors are zero).
__device__ __forceinline__ int QuantizeNonnegative(float value, unsigned int exponentByte, float dither)
{
    return __float2int_rd(__fmaf_rd(value, PowerOfTwo(kExponentBias - int(exponentByte)), dither));
}

// The exponent bytes of a decision's hands, following its action rows from units (see
// GpuQuantize.h).
template<typename Unit>
__device__ __forceinline__ unsigned char* ExponentBytes(Unit* units, U32 actions, U32 count)
{
    return (unsigned char*)(units + U64(actions) * count);
}

// One parent normalizes its entry policy once for all children and, while training,
// accumulates its cumulative strategy from the reach it propagates, re-encoding each
// hand with reach (see GpuQuantize.h) and leaving the others untouched. A thread carries
// two adjacent hands so twice the loads are in flight per warp; pair and value1 are zero
// when only hand h exists. A zero ActionCount keeps the general path, whose entry arrays
// live in local memory and whose loops stay rolled: unrolled to kMaxActions they triple the
// code and spill.
template<U32 ActionCount>
__device__ __forceinline__ void ReachNode(
    Node n,
    State state,
    U32 p,
    U32 h,
    U32 pair,
    float value0,
    float value1,
    const short* regrets,
    unsigned short* sums,
    float* scratch
)
{
    const U32 count = p ? state.hands[1] : state.hands[0];
    const U32 actions = ActionCount ? ActionCount : Count(n.info);
    constexpr U32 capacity = ActionCount ? ActionCount : kMaxActions;
    const bool accumulate = !state.evaluation;
    const bool accumulate0 = accumulate && value0 != 0.0f, accumulate1 = accumulate && value1 != 0.0f;
    // Reconverging the warp before its policy loads measured 0.5% faster.
    __syncwarp(__activemask());
    float entries0[capacity], entries1[capacity];
    float sums0[capacity], sums1[capacity];
    // Zero reach still clears every child slot; only its policy and sum loads are skipped.
    // The entry policy matches a hand's quantized regrets or, in evaluation, normalizes its
    // quantized sums. Both modes load the sums and select values rather than loads, so
    // every load is in flight at once.
    float positive0 = 0.0f, positive1 = 0.0f;
#pragma unroll(ActionCount ? ActionCount : 1)
    for (U32 a = 0; a < actions; ++a)
    {
        const U64 entry = n.strategy + U64(a) * count + h;
        const float s0 = value0 != 0.0f ? float(sums[entry]) : 0.0f;
        const float s1 = value1 != 0.0f ? float(sums[entry + 1]) : 0.0f;
        const float x0 = accumulate0 ? float(regrets[entry]) : s0;
        const float x1 = accumulate1 ? float(regrets[entry + 1]) : s1;
        entries0[a] = x0;
        entries1[a] = x1;
        sums0[a] = s0;
        sums1[a] = s1;
        positive0 += x0 > 0.0f ? x0 : 0.0f;
        positive1 += x1 > 0.0f ? x1 : 0.0f;
    }
    unsigned char* exponents = ExponentBytes(sums + n.strategy, actions, count);
    const U32 exponent0 = accumulate0 ? exponents[h] : 0u, exponent1 = accumulate1 ? exponents[h + 1] : 0u;
    float magnitude0 = 0.0f, magnitude1 = 0.0f;
#pragma unroll(ActionCount ? ActionCount : 1)
    for (U32 a = 0; a < actions; ++a)
    {
        float child0 = value0, child1 = value1;
        if (value0 != 0.0f)
            child0 *= MatchedProbability(entries0[a], positive0, actions);
        if (value1 != 0.0f)
            child1 *= MatchedProbability(entries1[a], positive1, actions);
        StorePair(scratch + U64(n.link + a) * state.stride + h, pair, child0, child1);
        if (accumulate)
        {
            sums0[a] = Dequantize(sums0[a], exponent0) + state.averageWeight * child0;
            sums1[a] = Dequantize(sums1[a], exponent1) + state.averageWeight * child1;
            magnitude0 = fmaxf(magnitude0, sums0[a]);
            magnitude1 = fmaxf(magnitude1, sums1[a]);
        }
    }
    if (accumulate0)
    {
        const U32 next = SumExponent(magnitude0);
        const float dither = Dither(U32(n.strategy + h), state.update);
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
            sums[n.strategy + U64(a) * count + h] = (unsigned short)QuantizeNonnegative(sums0[a], next, dither);
        exponents[h] = (unsigned char)next;
    }
    if (accumulate1)
    {
        const U32 next = SumExponent(magnitude1);
        const float dither = Dither(U32(n.strategy + h + 1), state.update);
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
            sums[n.strategy + U64(a) * count + h + 1] = (unsigned short)QuantizeNonnegative(sums1[a], next, dither);
        exponents[h + 1] = (unsigned char)next;
    }
}

// 48 registers keep about 40 warps per SM resident without spilling; 40 registers spill and
// lose 5%. Items tile their pass.lanes, two opponent hands per thread, like Backup; a tile
// compacts its pairs carrying reach to its first threads, since zero-reach hands cluster and
// only clear their children's rows.
__global__ void __launch_bounds__(kTileGroup, 5) Reach(GPU_ARGUMENTS)
{
    const State state = *statePointer;
    const U32 item = blockIdx.x, local = threadIdx.x, lane = blockIdx.y * blockDim.x + local;
    const Node n = nodes[pass.offset + item];
    // Only the opponent's reach is propagated, one row of stride floats per slot: terminals
    // read it and the opponent's cumulative strategy accumulates it. The updating player's
    // own reach is never needed.
    const U32 p = 1 - state.player, count = p ? state.hands[1] : state.hands[0], base = p ? state.hands[0] : 0;
    // Two adjacent hands per thread.
    const U32 h = 2 * lane;
    const bool inside = h < count;
    const U32 pair = h + 1 < count ? 1u : 0u;
    const float* source = scratch + U64(p ? n.reachSlot[1] : n.reachSlot[0]) * state.stride + h;
    // A thread without a pair loads hand h again as its second hand and discards it, so every
    // load of a derivation is unconditional.
    const U32 second = base + h + pair;
    float value0 = 0.0f, value1 = 0.0f;
    if (inside)
    {
        if (IsRoot(n.info))
        {
            // The game root's reach is the hands' weights.
            const float weight1 = hands[second].weight;
            value0 = hands[base + h].weight;
            value1 = pair ? weight1 : 0.0f;
        }
        else if (n.foldUtility != 0.0f)
        {
            // Region roots and the children that derive their reach (see Node) apply the dealt
            // card and the outcome probability to the chance parent's reach.
            const float2 entering = LoadPair(source, pair);
            const U64 mask0 = hands[base + h].mask, mask1 = hands[second].mask;
            value0 = (mask0 & n.board) ? 0.0f : entering.x * n.foldUtility;
            value1 = pair && !(mask1 & n.board) ? entering.y * n.foldUtility : 0.0f;
        }
        else
        {
            // The opponent's deeper decisions propagate its reach.
            const float2 entering = LoadPair(source, pair);
            value0 = entering.x;
            value1 = entering.y;
        }
    }
    if (Kind(n.info) != NodeKind::Decision || Actor(n.info) != p)
    {
        // A region root other than the opponent's decision stores the opponent's reach entering
        // it for its descendants; an opponent decision's children receive their own rows.
        if (inside)
            StorePair(scratch + U64(n.slot) * state.stride + h, pair, value0, value1);
        return;
    }
    const U32 actions = Count(n.info);
    // Pairs without reach clear their children's rows; the others move to the tile's first
    // threads, in any order: each pair's arithmetic depends only on its hands.
    __shared__ U32 liveCount;
    __shared__ U32 livePairs[kTileGroup];
    __shared__ float2 liveValues[kTileGroup];
    if (local == 0)
        liveCount = 0;
    const bool live = inside && (value0 != 0.0f || value1 != 0.0f);
    if (inside && !live)
    {
#pragma unroll 1
        for (U32 a = 0; a < actions; ++a)
            StorePair(scratch + U64(n.link + a) * state.stride + h, pair, 0.0f, 0.0f);
    }
    __syncthreads();
    const U32 ballot = __ballot_sync(0xffffffffu, live), before = ballot & ((1u << (local % kWarpSize)) - 1);
    U32 first = 0;
    if (before == 0 && live)
        first = atomicAdd(&liveCount, __popc(ballot));
    first = __shfl_sync(0xffffffffu, first, __ffs(ballot) - 1);
    if (live)
    {
        livePairs[first + __popc(before)] = h;
        liveValues[first + __popc(before)] = make_float2(value0, value1);
    }
    __syncthreads();
    if (local >= liveCount)
        return;
    const U32 liveH = livePairs[local], livePair = liveH + 1 < count ? 1u : 0u;
    const float2 v = liveValues[local];
    if (actions == 2)
        ReachNode<2>(n, state, p, liveH, livePair, v.x, v.y, regrets, sums, scratch);
    else if (actions == 3)
        ReachNode<3>(n, state, p, liveH, livePair, v.x, v.y, regrets, sums, scratch);
    else
        ReachNode<0>(n, state, p, liveH, livePair, v.x, v.y, regrets, sums, scratch);
}

// All lanes of the warp participate, including zero-padded tails.
__device__ __forceinline__ float WarpScan(float value, U32 lane)
{
    for (U32 offset = 1; offset < kWarpSize; offset *= 2)
    {
        const float previous = __shfl_up_sync(0xffffffffu, value, offset);
        if (lane >= offset)
            value += previous;
    }
    return value;
}

// The sum of the values of the lanes before this one.
__device__ __forceinline__ float WarpBefore(float value, U32 lane)
{
    const float previous = __shfl_up_sync(0xffffffffu, WarpScan(value, lane), 1);
    return lane ? previous : 0.0f;
}

// The float at byte offset of the staged rows.
__device__ __forceinline__ float Staged(const unsigned char* staged, U32 offset)
{
    return *(const float*)(staged + offset);
}

// Stores value at the shared address plus Offset bytes when index < length, one predicated store
// whose address stays in a register across the calls.
template<U32 Offset>
__device__ __forceinline__ void StoreBelow(U32 address, float value, U32 index, U32 length)
{
    asm volatile("{\n\t.reg .pred p;\n\tsetp.lt.u32 p, %2, %3;\n\t@p st.shared.f32 [%0+%4], %1;\n\t}" ::"r"(address),
                 "f"(value),
                 "r"(index),
                 "r"(length),
                 "n"(Offset)
                 : "memory");
}
template<U32 J, U32 Count>
__device__ __forceinline__ void StoreRun(U32 address, const float* x, float carried, U32 length)
{
    if constexpr (J < Count)
    {
        StoreBelow<4 * J>(address, x[J] + carried, J, length);
        StoreRun<J + 1, Count>(address, x, carried, length);
    }
}

// Four 16-bit byte offsets of a token group.
__device__ __forceinline__ void Unpack(uint2 t, U32* offsets)
{
    offsets[0] = t.x & 0xffffu;
    offsets[1] = t.x >> 16;
    offsets[2] = t.y & 0xffffu;
    offsets[3] = t.y >> 16;
}

// A warp scans a showdown's staged reach in rank order: each lane sums a contiguous segment of
// ranks, then adds the totals of the segments before it, so the scan takes a single warp scan.
// sums receives each rank's sum through it. The ranks' byte offsets in the staged reach come in
// four-rank groups per segment, group g of segment s at tokens[g * kWarpSize + s], ranks past a
// segment's end holding the zero slot's (see Plan::order): adding its +0 leaves a sum unchanged.
// Up to Groups groups the segment's sums stay in registers.
template<U32 Groups>
__device__ __forceinline__ void ScanSegments(U32 rankCount, const uint2* tokens, U32 lane, const unsigned char* staged, float* sums)
{
    const Segment bounds = LaneSegment(rankCount, lane);
    const U32 length = bounds.end - bounds.begin;
    float x[4 * Groups];
#pragma unroll
    for (U32 g = 0; g < Groups; ++g)
    {
        U32 offsets[4];
        Unpack(tokens[g * kWarpSize + lane], offsets);
#pragma unroll
        for (U32 c = 0; c < 4; ++c)
            x[4 * g + c] = Staged(staged, offsets[c]);
    }
    float sum = 0.0f;
#pragma unroll
    for (U32 j = 0; j < 4 * Groups; ++j)
    {
        sum += x[j];
        x[j] = sum;
    }
    const float carried = WarpBefore(sum, lane);
    StoreRun<0, 4 * Groups>(U32(__cvta_generic_to_shared(sums + bounds.begin)), x, carried, length);
}

// ScanSegments for segments of any length: sums pass through shared memory.
__device__ __forceinline__ void ScanSegmentsLong(U32 rankCount, const uint2* tokens, U32 lane, const unsigned char* staged, float* sums)
{
    const Segment bounds = LaneSegment(rankCount, lane);
    const U32 length = bounds.end - bounds.begin;
    const unsigned short* column = (const unsigned short*)(tokens + lane);
    float* out = sums + bounds.begin;
    float sum = 0.0f;
    for (U32 j = 0; j < length; ++j)
    {
        sum += Staged(staged, column[(j / 4) * (4 * kWarpSize) + j % 4]);
        out[j] = sum;
    }
    const float carried = WarpBefore(sum, lane);
    for (U32 j = 0; j < length; ++j)
        out[j] += carried;
}

__device__ __forceinline__ void TerminalScan(U32 rankCount, const uint2* tokens, U32 lane, const unsigned char* staged, float* sums)
{
    const U32 span = (rankCount + kWarpSize - 1) / kWarpSize;
    if (span <= 4)
        ScanSegments<1>(rankCount, tokens, lane, staged, sums);
    else if (span <= 8)
        ScanSegments<2>(rankCount, tokens, lane, staged, sums);
    else if (span <= 12)
        ScanSegments<3>(rankCount, tokens, lane, staged, sums);
    else
        ScanSegmentsLong(rankCount, tokens, lane, staged, sums);
}

// Stages the opponent's reach row of slot row into reach, one float4 group per thread (rows
// and their staged copies are padded to whole groups, see TerminalLayout), and returns whether
// it carries reach. A terminal without opponent reach leaves its values stale, and Backup
// substitutes zero from its flag.
__device__ __forceinline__ bool StageReach(State state, U32 row, U32 local, const float* scratch, float* reach)
{
    const U32 opponentCount = state.player ? state.hands[0] : state.hands[1];
    const float4* source = (const float4*)(scratch + U64(row) * state.stride);
    bool nonzero = false;
    for (U32 group = local; 4 * group <= opponentCount; group += kTerminalGroup)
    {
        const float4 r = StagedGroup(source, group, opponentCount - 4 * group);
        ((float4*)reach)[group] = r;
        nonzero |= AnyReach(r);
    }
    return __syncthreads_or(nonzero);
}

// StageReach for a fold's own row, into foldReach, together with its showdown sibling's row,
// into reach, so their loads overlap; returns whether each carries reach: bit 0 the fold's,
// bit 1 the showdown's.
__device__ __forceinline__ U32
StageRows(State state, U32 foldRow, U32 showdownRow, U32 local, const float* scratch, float* reach, float* foldReach)
{
    const U32 opponentCount = state.player ? state.hands[0] : state.hands[1];
    const float4* foldSource = (const float4*)(scratch + U64(foldRow) * state.stride);
    const float4* source = (const float4*)(scratch + U64(showdownRow) * state.stride);
    bool fold = false, showdown = false;
    for (U32 group = local; 4 * group <= opponentCount; group += kTerminalGroup)
    {
        const float4 f = StagedGroup(foldSource, group, opponentCount - 4 * group);
        const float4 r = StagedGroup(source, group, opponentCount - 4 * group);
        ((float4*)foldReach)[group] = f;
        ((float4*)reach)[group] = r;
        fold |= AnyReach(f);
        showdown |= AnyReach(r);
    }
    return (__syncthreads_or(fold) ? 1u : 0u) | (__syncthreads_or(showdown) ? 2u : 0u);
}

// One lane's card run of blocked reach: the running sums of the card's ranked holders' reach in
// rank order, whose byte offsets in the staged reach come in four-holder groups, group g at
// tokens[g * 2 * kWarpSize] (see Plan::order). lane packs the run's byte offset from runs, its
// length << 16 and the card << 24 (0xff without a card). groups, the most of the lane's warp,
// bounds the loop. With Fold, the holders' mass in a fold sibling's own row, staged at
// foldStaged, goes to masses[1 + card]; a lane without a card stores none.
template<bool Fold, U32 MaxGroups>
__device__ __forceinline__ void BlockerRun(
    U32 groups,
    U32 lane,
    const uint2* tokens,
    const unsigned char* staged,
    const unsigned char* foldStaged,
    float* runs,
    float* masses
)
{
    const U32 length = (lane >> 16) & 0xffu;
    float* run = (float*)((unsigned char*)runs + (lane & 0xffffu));
    float sum = 0.0f, foldMass = 0.0f;
    // Up to MaxGroups the groups' tokens load together before any sum; each group runs behind a
    // uniform test.
    uint2 loaded[MaxGroups ? MaxGroups : 1];
    if constexpr (MaxGroups != 0)
    {
#pragma unroll
        for (U32 g = 0; g < MaxGroups; ++g)
            loaded[g] = g < groups ? tokens[g * 2 * kWarpSize] : make_uint2(0u, 0u);
    }
    // Holders past the run point at the staged rows' zero slot: their +0 leaves the sums.
#pragma unroll
    for (U32 g = 0; g < (MaxGroups ? MaxGroups : groups); ++g)
    {
        if (g >= groups)
            break;
        U32 offsets[4];
        Unpack(MaxGroups ? loaded[g] : tokens[g * 2 * kWarpSize], offsets);
#pragma unroll
        for (U32 c = 0; c < 4; ++c)
        {
            sum += Staged(staged, offsets[c]);
            if (Fold)
                foldMass += Staged(foldStaged, offsets[c]);
            if (length > 4 * g + c)
                run[4 * g + c] = sum;
        }
    }
    if (Fold && (lane >> 24) != 0xffu)
        masses[1 + (lane >> 24)] = foldMass;
}

template<bool Fold>
__device__ __forceinline__ void BlockerRuns(
    U32 groups,
    U32 lane,
    const uint2* tokens,
    const unsigned char* staged,
    const unsigned char* foldStaged,
    float* runs,
    float* masses
)
{
    if (groups <= 6)
        BlockerRun<Fold, 6>(groups, lane, tokens, staged, foldStaged, runs, masses);
    else
        BlockerRun<Fold, 0>(groups, lane, tokens, staged, foldStaged, runs, masses);
}

// A fold's total opponent mass from the staged reach, summed by one warp.
__device__ __forceinline__ void FoldTotal(U32 opponentCount, U32 lane, const float* reach, float* total)
{
    // Every lane runs the same trip count, adding +0 past the row, so the warp reaches the
    // scan's shuffles converged.
    float sum = 0.0f;
    for (U32 first = 0; first < opponentCount; first += kWarpSize)
        sum += first + lane < opponentCount ? reach[first + lane] : 0.0f;
    sum = WarpScan(sum, lane);
    if (lane == kWarpSize - 1)
        total[0] = sum;
}

// Lists, in ascending order, the opponent hands of the staged reach that carry reach, into kept,
// and stores their number in total: each lane of one warp lists a contiguous segment.
__device__ __forceinline__ void ListReached(U32 opponentCount, U32 lane, const float* reach, U32* kept, U32* total)
{
    const Segment segment = LaneSegment(opponentCount, lane);
    const U32 begin = segment.begin, end = segment.end;
    U32 reached = 0;
    for (U32 other = begin; other < end; ++other)
        reached += reach[other] != 0.0f ? 1u : 0u;
    U32 next = U32(WarpBefore(float(reached), lane));
    for (U32 other = begin; other < end; ++other)
        if (reach[other] != 0.0f)
            kept[next++] = other;
    if (lane == kWarpSize - 1)
        total[0] = next;
}

// A fold's opponent masses from the staged reach: the total, then each card's holders' mass,
// as the CPU evaluator's per-card masses. The first warp sums the total while the others take
// the card masses.
__device__ __forceinline__ void FoldMasses(State state, U32 local, const U32* cards, const float* reach, float* masses)
{
    const U32 opponent = 1 - state.player, opponentCount = state.player ? state.hands[0] : state.hands[1];
    if (local < kWarpSize)
        FoldTotal(opponentCount, local, reach, masses);
    else
        for (U32 card = local - kWarpSize; card < 52; card += kTerminalGroup - kWarpSize)
        {
            float mass = 0.0f;
            for (U32 entry = cards[opponent * kCardListStride + card]; entry < cards[opponent * kCardListStride + card + 1]; ++entry)
                mass += reach[cards[entry]];
            masses[1 + card] = mass;
        }
}

// A hand's fold value from its fold's masses: the total less each held card's mass, adding
// back the identical opponent hand's reach (zero without one), which both cards' masses include.
__device__ __forceinline__ float FoldValue(const float* masses, U32 packedCards, float identicalReach, float scale, float payoff)
{
    const float mass = masses[0] - masses[1 + (packedCards & 0xffu)] - masses[1 + ((packedCards >> 8) & 0xffu)] + identicalReach;
    return (mass * scale) * payoff;
}

// A fold without a showdown sibling, from the staged opponent reach.
__device__ __forceinline__ void TerminalFold(
    Node n,
    State state,
    U32 local,
    const Hand* hands,
    const U32* cards,
    float* values,
    const float* reach,
    float* masses
)
{
    const U32 p = state.player;
    const U32 count = p ? state.hands[1] : state.hands[0], mineBase = p ? state.hands[0] : 0;
    FoldMasses(state, local, cards, reach, masses);
    __syncthreads();
    float* output = values + U64(n.slot) * state.stride;
    const float payoff = p == 0 ? n.utility[0] : -n.utility[0];
    for (U32 h = local; h < count; h += kTerminalGroup)
    {
        // The hand's first half: mask, value scale and packed cards (see Hand).
        const uint4 first = ((const uint4*)(hands + mineBase + h))[0];
        const U64 mask = U64(first.x) | (U64(first.y) << 32);
        const float scale = BitsFloat(first.z);
        const U32 identical = first.w >> 16;
        output[h] =
            (mask & n.board) || scale <= 0.0f ? 0.0f : FoldValue(masses, first.w, identical ? reach[identical - 1] : 0.0f, scale, payoff);
    }
}

// The stored regret after adding increment. Positive regrets are stored divided by
// positiveScale, so the discounts of the updates a node skipped are already in place;
// negative regrets halve once per skipped update.
__device__ __forceinline__ float UpdatedRegret(float old, float increment, float halving, State state)
{
    const float regret = (old > 0.0f ? old * state.positiveScale : old * halving) + increment;
    return regret > 0.0f ? regret * state.positiveInverse : regret * 0.5f;
}

// Advances an acting decision's stamp at a training update and returns the halving of its
// negative regrets for the updates it skipped. Stamps alternate between two halves: a pass
// reads the half its player's previous update wrote while writing the other, and every
// acting decision writes its entry, from the one writer thread.
__device__ __forceinline__ float AdvanceStamp(State state, U32 index, bool live, bool writer, U32* stamps)
{
    const U32 read = (state.update & 1u) * state.stampCount, write = ((state.update + 1u) & 1u) * state.stampCount;
    const U32 stamp = stamps[read + index];
    // A live node's store does not wait for the load.
    if (writer && live)
        stamps[write + index] = state.update;
    else if (writer)
        stamps[write + index] = stamp;
    return Halving(state.update - 1u - stamp);
}

// One hand of a decision whose only children are the fold and showdown the calling block
// evaluated: BackupNode's update without child loads. The decision has the given actor; entry0
// and exponents point at the hand's first regret and exponent byte (its second regret is count
// entries later), ditherIndex is the index of that first regret, and values points at the
// hand's value in the decision's scratch row; action foldAction is the fold.
__device__ __forceinline__ void BackupLeafPair(
    U32 actor,
    U32 ditherIndex,
    short* entry0,
    unsigned char* exponents,
    U32 count,
    State state,
    U32 foldAction,
    float fold,
    float showdown,
    float halving,
    float* values
)
{
    const float child0 = foldAction == 0 ? fold : showdown, child1 = foldAction == 0 ? showdown : fold;
    float value;
    if (actor != state.player)
        value = child0 + child1;
    else if (state.evaluation)
        value = child1 > child0 ? child1 : child0;
    else
    {
        short* entry1 = entry0 + count;
        const float r0 = float(entry0[0]), r1 = float(entry1[0]);
        const U32 exponent = exponents[0];
        const float positive = (r0 > 0.0f ? r0 : 0.0f) + (r1 > 0.0f ? r1 : 0.0f);
        value = child0 * MatchedProbability(r0, positive, 2) + child1 * MatchedProbability(r1, positive, 2);
        const float updated0 = UpdatedRegret(Dequantize(r0, exponent), child0 - value, halving, state);
        const float updated1 = UpdatedRegret(Dequantize(r1, exponent), child1 - value, halving, state);
        const U32 next = RegretExponent(fmaxf(fabsf(updated0), fabsf(updated1)));
        const float dither = Dither(ditherIndex, state.update);
        entry0[0] = (short)Quantize(updated0, next, dither);
        entry1[0] = (short)Quantize(updated1, next, dither);
        exponents[0] = (unsigned char)next;
    }
    values[0] = value;
}

// Registers are capped so twelve 128-thread groups, 48 warps, stay resident per SM. The first
// warp scans a showdown and the second sums a fold sibling's row while the last two build its
// blocker runs, one card per lane of its section's 2 * kWarpSize-entry lane table (see
// BuildSection in GpuPlan.cpp).
static_assert(kTerminalGroup == 4 * kWarpSize);
__global__ void __launch_bounds__(kTerminalGroup, 12) Terminal(GPU_ARGUMENTS)
{
    const State state = *statePointer;
    const U32 item = blockIdx.x, local = threadIdx.x;
    const U32 p = state.player, opponent = 1 - p;
    // Fixed indices keep the small State/Node arrays in registers.
    const U32 count = p ? state.hands[1] : state.hands[0], opponentCount = p ? state.hands[0] : state.hands[1];
    // The grid has exactly one block per item (see CudaExecutor's Dispatch).
    const Node n = nodes[pass.offset + item];
    // Thread 0 moves the record 1024 items ahead into L2 for its block: a fixed distance tuned on
    // an 84-SM GPU, where twelve resident blocks per SM make it about one wave.
    if (local == 0 && item + 1024 < pass.count)
        PrefetchL2(nodes + pass.offset + item + 1024);
    const NodeKind kind = Kind(n.info);
    const U32 row = Row(n.info);
    const U32 total = state.hands[0] + state.hands[1], mineBase = p ? state.hands[0] : 0;
    const U32 reachSlot = p ? n.reachSlot[0] : n.reachSlot[1];
    // Rows of the opponent's hands (see TerminalLayout): the staged rows are padded to float4 groups.
    extern __shared__ __align__(16) float reach[];
    float* prefix = reach + TerminalLayout::Padded(opponentCount + 1);
    float* blockers = prefix + opponentCount;
    const unsigned char* staged = (const unsigned char*)reach;
    // A fold's total and card masses, then a fold sibling's own row, past the blocker runs: a
    // live showdown sums the sibling's masses from that row while its scan and runs write the
    // prefix and run tables, and the hand loop reads all of them, the row after the fold's
    // values have replaced it in scratch.
    float* foldMasses = blockers + 2 * opponentCount;
    float* foldReach = reach + TerminalLayout::Padded(U32(foldMasses - reach) + 53);
    if (kind == NodeKind::Fold)
    {
        // A fold without a showdown sibling.
        const bool live = StageReach(state, reachSlot, local, scratch, reach);
        if (local == 0)
            flags[n.slot] = live ? 1u : 0u;
        if (live)
            TerminalFold(n, state, local, hands, cards, scratch, reach, foldMasses);
        return;
    }
    // A showdown evaluates its fold sibling from the same opponent reach row when the
    // updating player folds there, otherwise from the masses of the fold's own row, and backs
    // the parent up when the two are its only children (see Node).
    const bool sibling = n.fold != kNoIndex;
    // A showdown with a fold sibling carries the parent fields the block reads (see Node).
    const bool ownFold = sibling && Actor(n.info) == p;
    const bool backup = sibling && Count(n.info) == 2;
    // As in BackupNode, an acting parent's regrets train and its stamp advances at every training
    // update.
    const bool trainParent = backup && ownFold && !state.evaluation;
    // Siblings hold consecutive slots, so the fold is action 0 when its slot comes first.
    const U32 foldAction = n.fold < n.slot ? 0u : 1u;
    bool liveFold, live;
    if (sibling && !ownFold)
    {
        // The fold's own row, propagated by Reach, stages with the showdown's row; the hand
        // loop below takes the fold's values from its masses, which a live showdown's second
        // warp and blocker runs sum along.
        const U32 any = StageRows(state, n.fold, reachSlot, local, scratch, reach, foldReach);
        liveFold = (any & 1u) != 0;
        live = (any & 2u) != 0;
        if (liveFold && !live)
            FoldMasses(state, local, cards, foldReach, foldMasses);
    }
    else
    {
        live = StageReach(state, reachSlot, local, scratch, reach);
        liveFold = ownFold && live;
    }
    const bool liveParent = live || liveFold, foldFromRow = liveFold && !ownFold;
    const float halving = trainParent ? AdvanceStamp(state, n.stamp, liveParent, local == 0, stamps) : 1.0f;
    if (local == 0)
    {
        if (backup)
            flags[n.link] = liveParent ? 1u : 0u;
        else
        {
            flags[n.slot] = live ? 1u : 0u;
            if (sibling)
                flags[n.fold] = liveFold ? 1u : 0u;
        }
    }
    if (!liveParent)
        return;
    // A live training parent's regrets are read only after the showdown; their DRAM latency
    // overlaps the scan and blocker runs instead. Its two regret rows and exponent bytes are
    // contiguous, 5 * count bytes: one thread per 128-byte line, the last covering the final byte.
    const U32 legal = (n.rankCounts >> (16 * opponent)) & 0xffffu;
    if (trainParent && 128 * local < 5 * count + 127)
        PrefetchL2((const unsigned char*)(regrets + n.strategy) + (128 * local < 5 * count ? 128 * local : 5 * count - 1));
    if (live && kind == NodeKind::Showdown)
    {
        // The shuffle tells the compiler that the warp index is uniform in the warp, so its
        // branches keep shared addresses in uniform registers.
        const U32 warp = __shfl_sync(0xffffffffu, local / kWarpSize, 0), lane = local % kWarpSize;
        // The row's section for the opponent's hands (see Plan::order): a header, the scan
        // tokens, the lane table and the run tokens.
        const uint4* section = (const uint4*)(order + U64(row) * state.orderPitch + (opponent ? state.orderSection : 4 * total));
        // The first warp scans the reach in rank order; the second sums a fold sibling's own row.
        if (warp == 0)
            TerminalScan(legal, (const uint2*)(section + 1), lane, staged, prefix);
        else if (warp == 1)
        {
            if (foldFromRow)
                FoldTotal(opponentCount, lane, foldReach, foldMasses);
        }
        else
        {
            // Showdowns accumulate each card's blocked reach in rank order once per node, as the
            // CPU evaluator's per-card masses do, instead of rescanning blockers per hand. The
            // last two warps build them while the first scans, one card per lane, together with
            // a fold sibling's card masses from its own row.
            const uint4 header = section[0];
            const U32 runLane = local - 2 * kWarpSize;
            const U32 laneRun = ((const U32*)(section + header.y))[runLane];
            const U32 groups = warp == 2 ? header.x & 0xffu : (header.x >> 8) & 0xffu;
            const uint2* tokens = (const uint2*)(section + header.z) + runLane;
            if (foldFromRow)
                BlockerRuns<true>(groups, laneRun, tokens, staged, (const unsigned char*)foldReach, blockers, foldMasses);
            else
                BlockerRuns<false>(groups, laneRun, tokens, staged, staged, blockers, foldMasses);
        }
    }
    // A forced runout's hand loop visits only opponent hands with reach, listed in the prefix
    // space with their number in the run table's, both unused by runouts: a zero-reach hand's
    // terms would add exact zeros.
    U32* kept = (U32*)prefix;
    U32* keptCount = (U32*)blockers;
    if (live && kind == NodeKind::ForcedRunout && local < kWarpSize)
        ListReached(opponentCount, local, reach, kept, keptCount);
    __syncthreads();
    const float win = p == 0 ? n.utility[0] : -n.utility[2];
    const float tie = p == 0 ? n.utility[1] : -n.utility[1];
    const float loss = p == 0 ? n.utility[2] : -n.utility[0];
    const float foldPayoff = p == 0 ? n.foldUtility : -n.foldUtility;
    // As in the CPU evaluator, zero-tie showdowns need only win/loss masses; a fold from
    // the same row needs the mass. Loss masses are the ranked mass less the forward prefix.
    const bool needMass = tie != 0.0f || ownFold;
    const float rankedMass = live && legal ? prefix[legal - 1] : 0.0f;
    const float winScale = win - tie, lossScale = loss - tie;
    // Per-thread pointers advance with the hand.
    const Hand* hand = hands + mineBase + local;
    const uint4* handOffsets = (const uint4*)(order + U64(row) * state.orderPitch) + mineBase + local;
    // Values replace reach rows in scratch once the block has staged the reach it reads, a
    // backed-up parent's replacing the showdown's.
    float* handOutput = scratch + U64(backup ? n.link : n.slot) * state.stride + local;
    float* handFoldOutput = scratch + U64(sibling ? n.fold : n.slot) * state.stride + local;
    short* parentRegrets = regrets + n.strategy;
    unsigned char* parentExponents = ExponentBytes(parentRegrets, 2, count);
    for (U32 h = local; h < count; h += kTerminalGroup,
             hand += kTerminalGroup,
             handOffsets += kTerminalGroup,
             handOutput += kTerminalGroup,
             handFoldOutput += kTerminalGroup)
    {
        // The hand's first half: mask, value scale and packed cards (see Hand).
        const uint4 first = *(const uint4*)hand;
        // A live showdown's offsets load with the hand's record rather than after its board
        // test; a blocked hand's entry goes unused.
        const uint4 offsets = live && kind != NodeKind::ForcedRunout ? *handOffsets : make_uint4(0u, 0u, 0u, 0u);
        const U64 mask = U64(first.x) | (U64(first.y) << 32);
        const float scale = BitsFloat(first.z);
        const U32 identical = first.w >> 16;
        float showdown = 0.0f, fold = 0.0f;
        const bool open = !(mask & n.board) && scale > 0.0f;
        if (live && open)
        {
            float mass = 0.0f, wins = 0.0f, losses = 0.0f;
            if (kind == NodeKind::ForcedRunout)
            {
                // This player's hands are contiguous per opponent hand in its layout of the row
                // (see OutcomesBuffer), so a warp's loads coalesce; eight independent loads per
                // step overlap the row's latency while the sums keep their opponent order over the
                // opponent hands with reach.
                const U64 pairs = U64(state.hands[0]) * state.hands[1];
                const U32* table = outcomes + (p == 0 ? U64(state.outcomeRows) * pairs : 0) + U64(row) * pairs + h;
#define GPU_RUNOUT_STEP(countsValue, index)                               \
    {                                                                     \
        const U32 counts = (countsValue);                                 \
        /* Overlapping pairs hold the sentinel and contribute nothing. */ \
        const float r = counts == 0xffffffffu ? 0.0f : reach[index];      \
        mass += r;                                                        \
        wins += r * float(p == 0 ? counts & 0xffffu : counts >> 16);      \
        losses += r * float(p == 0 ? counts >> 16 : counts & 0xffffu);    \
    }
                const U32 reached = keptCount[0];
                U32 k = 0;
                for (; k + 8 <= reached; k += 8)
                {
                    const U32 o0 = kept[k], o1 = kept[k + 1], o2 = kept[k + 2], o3 = kept[k + 3];
                    const U32 o4 = kept[k + 4], o5 = kept[k + 5], o6 = kept[k + 6], o7 = kept[k + 7];
                    const U32 c0 = table[U64(o0) * count], c1 = table[U64(o1) * count];
                    const U32 c2 = table[U64(o2) * count], c3 = table[U64(o3) * count];
                    const U32 c4 = table[U64(o4) * count], c5 = table[U64(o5) * count];
                    const U32 c6 = table[U64(o6) * count], c7 = table[U64(o7) * count];
                    GPU_RUNOUT_STEP(c0, o0)
                    GPU_RUNOUT_STEP(c1, o1)
                    GPU_RUNOUT_STEP(c2, o2)
                    GPU_RUNOUT_STEP(c3, o3)
                    GPU_RUNOUT_STEP(c4, o4)
                    GPU_RUNOUT_STEP(c5, o5)
                    GPU_RUNOUT_STEP(c6, o6)
                    GPU_RUNOUT_STEP(c7, o7)
                }
                for (; k < reached; ++k)
                    GPU_RUNOUT_STEP(table[U64(kept[k]) * count], kept[k])
#undef GPU_RUNOUT_STEP
                // A runout record's win and loss are the scales of its counts (see Node).
                showdown = (tie * mass + wins * win + losses * loss) * scale;
            }
            else
            {
                // Where the hand's masses are, which depends only on the board and ranges, as 16-bit
                // byte offsets from the staged reach (see Plan::order): its win mass | the ranked
                // mass not stronger than it in the forward prefix, per held card its blocked reach
                // below | through its rank in the card's run, and both cards' run totals.
                const float winMass = Staged(staged, offsets.x & 0xffffu);
                const float lossMass = rankedMass - Staged(staged, offsets.x >> 16);
                const float all0 = Staged(staged, offsets.w & 0xffffu), all1 = Staged(staged, offsets.w >> 16);
                const float below0 = Staged(staged, offsets.y & 0xffffu), through0 = Staged(staged, offsets.y >> 16);
                const float below1 = Staged(staged, offsets.z & 0xffffu), through1 = Staged(staged, offsets.z >> 16);
                if (needMass)
                {
                    mass = rankedMass - (all0 + all1);
                    if (identical)
                        mass += reach[identical - 1];
                }
                wins = winMass - (below0 + below1);
                // A card's list has at most 51 entries, so all - through loses only ulps of that
                // card's mass.
                losses = lossMass - ((all0 - through0) + (all1 - through1));
                showdown = (tie * mass + winScale * wins + lossScale * losses) * scale;
                if (ownFold)
                    fold = (mass * scale) * foldPayoff;
            }
        }
        if (foldFromRow && open)
            fold = FoldValue(foldMasses, first.w, identical ? foldReach[identical - 1] : 0.0f, scale, foldPayoff);
        if (!backup)
        {
            if (live)
                *handOutput = showdown;
            if (liveFold)
                *handFoldOutput = fold;
        }
        else
            BackupLeafPair(
                Actor(n.info),
                U32(n.strategy + h),
                parentRegrets + h,
                parentExponents + h,
                count,
                state,
                foldAction,
                fold,
                showdown,
                halving,
                handOutput
            );
    }
}

// Small decisions retain entry data in registers and issue every load before the first
// store so latencies overlap; a zero ActionCount keeps the general path, whose entry
// arrays live in local memory and whose loops stay rolled (see ReachNode). A thread
// carries two adjacent hands so twice the loads are in flight per warp; pair is zero when
// only hand h exists. Child values load with their flags, before the node's liveness is
// known, and dead children's are replaced by zero, so the flag and value latencies
// overlap; regrets wait for liveness, which spares pruned nodes their DRAM traffic. A node
// is live when any child subtree carries opponent reach; pruned nodes have zero values and
// keep their regrets, whose skipped discounts the stamps apply at the next live update. An
// update re-encodes each hand's regrets (see GpuQuantize.h).
template<U32 ActionCount>
__device__ __forceinline__ void BackupNode(Node n, State state, U32 h, U32 pair, short* regrets, float* values, U32* flags, U32* stamps)
{
    const U32 count = state.player ? state.hands[1] : state.hands[0];
    const U32 actions = ActionCount ? ActionCount : Count(n.info);
    const bool acting = Kind(n.info) == NodeKind::Decision && Actor(n.info) == state.player;
    const bool training = acting && !state.evaluation;
    constexpr U32 capacity = ActionCount ? ActionCount : kMaxActions;
    U32 lives[capacity];
    float entryRegrets0[capacity], entryRegrets1[capacity];
    float childValues0[capacity], childValues1[capacity];
    U32 live = 0;
    if (ActionCount && acting)
    {
        // An inlined child (see Node) has no Backup item in this update: its children's flags
        // and values load in its place, all loads before any sum, and form its flag and value
        // as its own Backup would, whether any child is live and the live children's values
        // summed in child order. Other children load as one source each.
        U32 sourceFlags[capacity][3];
        float sources0[capacity][3], sources1[capacity][3];
        U32 descriptors[capacity];
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            descriptors[a] = a == 0 ? U32(n.board) : a == 1 ? U32(n.board >> 32) : n.fold;
            const bool inlined = descriptors[a] != kNoIndex;
            const U32 first = inlined ? descriptors[a] & 0x3fffffffu : n.link + a, sources = inlined ? descriptors[a] >> 30 : 1u;
            for (U32 k = 0; k < 3; ++k)
            {
                const bool present = k < sources;
                const float2 source = present ? LoadPair(values + U64(first + k) * state.stride + h, pair) : make_float2(0.0f, 0.0f);
                sourceFlags[a][k] = present ? flags[first + k] : 0u;
                sources0[a][k] = source.x;
                sources1[a][k] = source.y;
            }
        }
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            if (descriptors[a] == kNoIndex)
            {
                lives[a] = sourceFlags[a][0];
                childValues0[a] = sources0[a][0];
                childValues1[a] = sources1[a][0];
            }
            else
            {
                // Absent sources add +0, which leaves the sum unchanged: a sum from +0 is never -0.
                lives[a] = 0;
                childValues0[a] = childValues1[a] = 0.0f;
                for (U32 k = 0; k < 3; ++k)
                {
                    lives[a] |= sourceFlags[a][k];
                    childValues0[a] += sourceFlags[a][k] ? sources0[a][k] : 0.0f;
                    childValues1[a] += sourceFlags[a][k] ? sources1[a][k] : 0.0f;
                }
            }
            live |= lives[a];
        }
    }
    else
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            const U32 slot = n.link + a;
            const U32 flag = flags[slot];
            lives[a] = flag;
            live |= flag;
            const float2 source = LoadPair(values + U64(slot) * state.stride + h, pair);
            childValues0[a] = source.x;
            childValues1[a] = source.y;
        }
    if (h == 0)
        flags[n.slot] = live;
    const float halving = training ? AdvanceStamp(state, n.stamp, live != 0, h == 0, stamps) : 1.0f;
    float* output = values + U64(n.slot) * state.stride + h;
    if (!live)
    {
        if (IsRoot(n.info))
            StorePair(output, pair, 0.0f, 0.0f);
        return;
    }
    // Reuse the entry-policy denominator for value backup and the regret update.
    float positive0 = 0.0f, positive1 = 0.0f;
    unsigned char* exponents = ExponentBytes(regrets + n.strategy, actions, count);
    U32 exponent0 = 0, exponent1 = 0;
    if (training)
    {
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            const short* entry = regrets + n.strategy + U64(a) * count + h;
            const float r0 = float(entry[0]), r1 = pair ? float(entry[1]) : 0.0f;
            entryRegrets0[a] = r0;
            entryRegrets1[a] = r1;
            positive0 += r0 > 0.0f ? r0 : 0.0f;
            positive1 += r1 > 0.0f ? r1 : 0.0f;
        }
        exponent0 = exponents[h];
        exponent1 = pair ? exponents[h + 1] : 0u;
    }
    float value0 = 0.0f, value1 = 0.0f;
#pragma unroll(ActionCount ? ActionCount : 1)
    for (U32 a = 0; a < actions; ++a)
    {
        const float child0 = lives[a] ? childValues0[a] : 0.0f, child1 = lives[a] ? childValues1[a] : 0.0f;
        // Only the regret update reads them back. The general path's local-memory arrays already
        // hold the loaded values, so it stores only dead children's zeros.
        if (ActionCount || (training && !lives[a]))
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
            value0 += child0 * MatchedProbability(entryRegrets0[a], positive0, actions);
            if (pair)
                value1 += child1 * MatchedProbability(entryRegrets1[a], positive1, actions);
        }
        else
        {
            value0 += child0;
            value1 += child1;
        }
    }
    StorePair(output, pair, value0, value1);
    if (training)
    {
        float magnitude0 = 0.0f, magnitude1 = 0.0f;
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            entryRegrets0[a] = UpdatedRegret(Dequantize(entryRegrets0[a], exponent0), childValues0[a] - value0, halving, state);
            magnitude0 = fmaxf(magnitude0, fabsf(entryRegrets0[a]));
            if (pair)
            {
                entryRegrets1[a] = UpdatedRegret(Dequantize(entryRegrets1[a], exponent1), childValues1[a] - value1, halving, state);
                magnitude1 = fmaxf(magnitude1, fabsf(entryRegrets1[a]));
            }
        }
        const U32 next0 = RegretExponent(magnitude0), next1 = RegretExponent(magnitude1);
        const float dither0 = Dither(U32(n.strategy + h), state.update), dither1 = Dither(U32(n.strategy + h + 1), state.update);
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            short* entry = regrets + n.strategy + U64(a) * count + h;
            entry[0] = (short)Quantize(entryRegrets0[a], next0, dither0);
            if (pair)
                entry[1] = (short)Quantize(entryRegrets1[a], next1, dither1);
        }
        exponents[h] = (unsigned char)next0;
        if (pair)
            exponents[h + 1] = (unsigned char)next1;
    }
}

// A chance node sums its children, whose slots are consecutive from its link (see
// Plan), so eight children's flags and values are loaded per round with no dependent
// slot lookup; the sums keep child order. Launches of a few such nodes end an update.
__device__ __forceinline__ void BackupChance(Node n, State state, U32 h, U32 pair, float* values, U32* flags)
{
    U32 live = 0;
    float value0 = 0.0f, value1 = 0.0f;
    const U32 children = Count(n.info);
#pragma unroll 1
    for (U32 a = 0; a < children; a += 8)
    {
        U32 flag[8];
        float loaded0[8], loaded1[8];
        for (U32 k = 0; k < 8; ++k)
        {
            const bool present = a + k < children;
            const float2 source = present ? LoadPair(values + U64(n.link + a + k) * state.stride + h, pair) : make_float2(0.0f, 0.0f);
            flag[k] = present ? flags[n.link + a + k] : 0u;
            loaded0[k] = source.x;
            loaded1[k] = source.y;
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
    float* output = values + U64(n.slot) * state.stride + h;
    if (!live && !IsRoot(n.info))
        return;
    StorePair(output, pair, value0, value1);
}

// 48 registers keep 42 warps per SM resident for both hand-tile sizes instead of 36.
__global__ void __launch_bounds__(kTileGroup, 5) Backup(GPU_ARGUMENTS)
{
    const State state = *statePointer;
    const U32 item = blockIdx.x, h = 2 * (blockIdx.y * blockDim.x + threadIdx.x);
    const U32 count = state.player ? state.hands[1] : state.hands[0];
    // The grid has one block row per item (see CudaExecutor's Dispatch); a last hand tile can
    // run past the hands.
    if (h >= count)
        return;
    const U32 pair = h + 1 < count ? 1u : 0u;
    const Node n = nodes[pass.offset + item];
    const U32 actions = Count(n.info);
    if (Kind(n.info) == NodeKind::Chance)
        BackupChance(n, state, h, pair, scratch, flags);
    else if (actions == 2)
        BackupNode<2>(n, state, h, pair, regrets, scratch, flags, stamps);
    else if (actions == 3)
        BackupNode<3>(n, state, h, pair, regrets, scratch, flags, stamps);
    else
        BackupNode<0>(n, state, h, pair, regrets, scratch, flags, stamps);
}

__global__ void Outcomes(GPU_ARGUMENTS)
{
    const State state = *statePointer;
    const U64 index = blockIdx.x * blockDim.x + threadIdx.x, pairs = U64(state.hands[0]) * state.hands[1];
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
                const U64 offset = U64(rankRow) * (state.hands[0] + state.hands[1]);
                const U32 a = ranks[offset + h0], b = ranks[offset + state.hands[0] + h1];
                wins += a > b;
                losses += a < b;
            }
        }
    }
    outcomes[index] = outcomes[transposed] = wins | (losses << 16);
}
