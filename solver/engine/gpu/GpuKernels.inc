// CUDA kernels of the FP32 DCFR passes, included once by CudaExecutor.cu.
using namespace solver::engine::gpu;
// Every kernel takes the buffers in BufferIndex order, the last holding the update's State, then
// the pass by value.
#define GPU_ARGUMENTS                                                                                                               \
    const Node *nodes, const Hand *hands, const float *weights, const unsigned short *ranks, const int *runouts, const U32 *order,  \
        const U32 *cards, U32 *outcomes, short *regrets, unsigned short *sums, float *scratch, U32 *flags, U32 *masks, U32 *stamps, \
        const State *statePointer, Pass pass
// numerator / divisor for regret matching's operands: quantized integers, the divisor at most
// kMaxActions * 65535 < 2^21. For such normal operands CUDA's division fast path, a refined
// reciprocal and a residual correction, is the correctly rounded quotient, and zero numerators
// give +0; `/` would add a range check whose slow-path call every warp with a zero divisor
// (a result discarded below) takes. A hand's actions share the divisor and so its refined
// reciprocal, RegretReciprocal.
__device__ __forceinline__ float RegretReciprocal(float divisor)
{
    float estimate;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(estimate) : "f"(divisor));
    return __fmaf_rn(estimate, __fmaf_rn(-divisor, estimate, 1.0f), estimate);
}

__device__ __forceinline__ float RegretRatio(float numerator, float divisor, float reciprocal)
{
    const float quotient = __fmaf_rn(numerator, reciprocal, 0.0f);
    return __fmaf_rn(reciprocal, __fmaf_rn(-divisor, quotient, numerator), quotient);
}

// Regret matching: positive is the hand's positive-regret sum over all actions and reciprocal
// its RegretReciprocal. The probability is finite, so a zero reach times it stays zero.
__device__ __forceinline__ float MatchedProbability(float regret, float positive, float reciprocal, U32 actions)
{
    return positive > 0.0f ? RegretRatio(regret > 0.0f ? regret : 0.0f, positive, reciprocal) : 1.0f / float(actions);
}

// Training state streams through the L2 once per update while a batch's passes reuse its scratch
// rows: state accesses are cache-streaming (evict-first), so the L2 replaces state lines before rows.
template<typename T>
__device__ __forceinline__ T LoadStreamed(const T* address)
{
    return __ldcs(address);
}
template<typename T>
__device__ __forceinline__ void StoreStreamed(T* address, T value)
{
    __stcs(address, value);
}

// A work item's record, which only its block, or the blocks of its hand tiles, read, once per
// update: a cache-streaming load leaves its line the L2's first to replace (on wide ranges, normal
// loads for tiled items measured no faster).
__device__ __forceinline__ Node LoadRecord(const Node* record)
{
    union
    {
        uint4 words[sizeof(Node) / sizeof(uint4)];
        Node node;
    } loaded;
#pragma unroll
    for (U32 k = 0; k < sizeof(Node) / sizeof(uint4); ++k)
        loaded.words[k] = LoadStreamed((const uint4*)record + k);
    return loaded.node;
}

// Hands h and h + 1 of a scratch row as one access: State::stride, a multiple of four, keeps rows
// 16-byte aligned, so an even h is 8-byte aligned, and holds a lane past an odd hand count.
// Without a pair, hand h + 1 is that lane, which loads read as zero and stores fill with the
// zero its thread computes for it; readers ignore lanes past the hands (see StagedGroup).
__device__ __forceinline__ float2 LoadPair(const float* row, U32 pair)
{
    const float2 loaded = *(const float2*)row;
    return make_float2(loaded.x, pair ? loaded.y : 0.0f);
}

__device__ __forceinline__ void StorePair(float* row, float value0, float value1)
{
    *(float2*)row = make_float2(value0, value1);
}

// A row's float4 group with the lanes past its hands, which hold stale scratch, zeroed; valid
// is the number of hands from the group's first. A group past the row's last hand loads nothing.
__device__ __forceinline__ float4 StagedGroup(const float4* source, U32 group, U32 valid)
{
    const float4 r = valid > 0 ? source[group] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return make_float4(r.x, valid > 1 ? r.y : 0.0f, valid > 2 ? r.z : 0.0f, valid > 3 ? r.w : 0.0f);
}

// value when keep's low bit is set, else +0: a bitwise select, which keeps the value's load
// unconditional instead of predicated on the mask's arrival.
__device__ __forceinline__ float Keep(float value, U32 keep)
{
    return BitsFloat(FloatBits(value) & (0u - (keep & 1u)));
}

// LoadPair of a reach row's pair with the pair's mask bit in keep's low bit (see MasksBuffer): a
// dead pair's stale lanes read as the +0 Reach stores for a live pair's zero hand.
__device__ __forceinline__ float2 LoadMaskedPair(const float* row, U32 pair, U32 keep)
{
    const float2 loaded = LoadPair(row, pair);
    return make_float2(Keep(loaded.x, keep), Keep(loaded.y, keep));
}

// StagedGroup of a reach row whose mask words are at mask (see MasksBuffer): a dead pair's stale
// lanes read as +0.
__device__ __forceinline__ float4 MaskedGroup(const float4* source, const U32* mask, U32 group, U32 valid)
{
    const float4 r = StagedGroup(source, group, valid);
    // The bits of the group's pairs 2 group and 2 group + 1.
    const U32 keep = mask[group / (kWarpSize / 2)] >> 2 * (group % (kWarpSize / 2));
    return make_float4(Keep(r.x, keep), Keep(r.y, keep), Keep(r.z, keep >> 1), Keep(r.w, keep >> 1));
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

// Moves the node records of the next launch in the pass's stream (see Pass) into L2, where its
// blocks find them instead of waiting for DRAM: the last blocks of the pass's first hand tile
// issue one 128-byte line each, the last line clamped to the final byte, or, when the pass has
// fewer blocks than lines, every block several.
__device__ __forceinline__ void PrefetchNextNodes(const Node* nodes, const Pass& pass, U32 local)
{
    const U32 bytes = pass.nextCount * U32(sizeof(Node)), lines = bytes ? (bytes + 127) / 128 + 1 : 0;
    const U32 skip = pass.count > lines ? pass.count - lines : 0;
    if (blockIdx.y != 0 || blockIdx.x < skip)
        return;
    const unsigned char* first = (const unsigned char*)(nodes + pass.next);
    for (U32 line = blockIdx.x - skip + local * pass.count; line < lines; line += blockDim.x * pass.count)
        PrefetchL2(first + (128 * line < bytes ? 128 * line : bytes - 1));
}

// Programmatic dependent launches (see CudaExecutor's Dispatch) start a pass's blocks while its
// predecessors finish: until this wait a block reads only State and the node records, which no
// pass writes.
__device__ __forceinline__ void WaitForPredecessors()
{
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Quantize (see GpuQuantize.h) of a value that is not negative: one fused multiply-add rounded
// down sums the exact scaled value and the dither, whose floor Quantize returns (a scaled value's
// own rounding only matters below 2^-126, where both floors are zero).
__device__ __forceinline__ int QuantizeNonnegative(float value, unsigned int exponentByte, float dither)
{
    return __float2int_rd(__fmaf_rd(value, PowerOfTwo(kExponentBias - int(exponentByte)), dither));
}

// The exponent bytes of a decision's hands, following its action rows of row units from units
// (see GpuQuantize.h).
template<typename Unit>
__device__ __forceinline__ unsigned char* ExponentBytes(Unit* units, U32 actions, U32 row)
{
    return (unsigned char*)(units + U64(actions) * row);
}

// A pair word's two entries (see StateRow): regrets or, in Averaging instances, cumulative
// strategies: nonnegative integers, which regret matching in evaluation (see Terminal) normalizes
// into the average strategy.
template<bool Averaging>
__device__ __forceinline__ float2 MatchedPair(U32 word)
{
    return Averaging ? make_float2(float(word & 0xffffu), float(word >> 16)) : make_float2(float(short(word)), float(int(word) >> 16));
}

// One parent normalizes its entry policy once for all children and, while training,
// accumulates its cumulative strategy from the reach it propagates. A thread carries two adjacent
// hands, whose entries and exponent bytes share words (see StateRow), and re-encodes both (see
// GpuQuantize.h): a hand without reach, value1 when only hand h exists, keeps its sums, which
// re-encoding at its own exponent byte reproduces. A zero ActionCount keeps the general path,
// whose entry arrays live in local memory and whose loops stay rolled: unrolled to kMaxActions
// they triple the code and spill.
template<U32 ActionCount>
__device__ __forceinline__ void ReachNode(
    Node n,
    State state,
    U32 p,
    U32 h,
    float value0,
    float value1,
    const short* regrets,
    unsigned short* sums,
    float* scratch
)
{
    const U32 count = p ? state.hands[1] : state.hands[0], row = StateRow(count);
    const U32 actions = ActionCount ? ActionCount : Count(n.info);
    constexpr U32 capacity = ActionCount ? ActionCount : kMaxActions;
    const bool accumulate = state.evaluation == Evaluation::None;
    // Reconverging the warp before its policy loads measured 0.5% faster.
    __syncwarp(__activemask());
    float entries0[capacity], entries1[capacity];
    float sums0[capacity], sums1[capacity];
    // Zero reach still clears every child slot. The entry policy matches a hand's quantized
    // regrets or, in evaluation, normalizes its quantized sums. Both modes load the sums and
    // select values rather than loads, so every load is in flight at once. The pair's entries
    // share a word (see StateRow).
    unsigned short* units = sums + StateOffset(n.strategy) + h;
    const short* regretUnits = regrets + StateOffset(n.strategy) + h;
    float positive0 = 0.0f, positive1 = 0.0f;
#pragma unroll(ActionCount ? ActionCount : 1)
    for (U32 a = 0; a < actions; ++a)
    {
        const U32 s = LoadStreamed((const U32*)(units + U64(a) * row));
        const U32 r = accumulate ? LoadStreamed((const U32*)(regretUnits + U64(a) * row)) : 0u;
        const float2 sum = MatchedPair<true>(s), entry = accumulate ? MatchedPair<false>(r) : sum;
        entries0[a] = entry.x;
        entries1[a] = entry.y;
        sums0[a] = sum.x;
        sums1[a] = sum.y;
        positive0 += entry.x > 0.0f ? entry.x : 0.0f;
        positive1 += entry.y > 0.0f ? entry.y : 0.0f;
    }
    unsigned char* exponents = ExponentBytes(sums + StateOffset(n.strategy), actions, row);
    const U32 exponentPair = accumulate ? LoadStreamed((const unsigned short*)(exponents + h)) : 0u;
    const U32 exponent0 = exponentPair & 0xffu, exponent1 = exponentPair >> 8;
    const float reciprocal0 = RegretReciprocal(positive0), reciprocal1 = RegretReciprocal(positive1);
    float* children = scratch + U64(n.link) * state.stride + h;
    float magnitude0 = 0.0f, magnitude1 = 0.0f;
#pragma unroll(ActionCount ? ActionCount : 1)
    for (U32 a = 0; a < actions; ++a)
    {
        const float child0 = value0 * MatchedProbability(entries0[a], positive0, reciprocal0, actions);
        const float child1 = value1 * MatchedProbability(entries1[a], positive1, reciprocal1, actions);
        StorePair(children + a * state.stride, child0, child1);
        if (accumulate)
        {
            sums0[a] = Dequantize(sums0[a], exponent0) + state.averageWeight * child0;
            sums1[a] = Dequantize(sums1[a], exponent1) + state.averageWeight * child1;
            magnitude0 = fmaxf(magnitude0, sums0[a]);
            magnitude1 = fmaxf(magnitude1, sums1[a]);
        }
    }
    if (accumulate)
    {
        const U32 next0 = value0 != 0.0f ? SumExponent(magnitude0) : exponent0;
        const U32 next1 = value1 != 0.0f ? SumExponent(magnitude1) : exponent1;
        const float dither0 = Dither(U32(n.strategy) + h, state.update), dither1 = Dither(U32(n.strategy) + h + 1, state.update);
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
            StoreStreamed(
                (U32*)(units + U64(a) * row),
                U32(QuantizeNonnegative(sums0[a], next0, dither0)) | U32(QuantizeNonnegative(sums1[a], next1, dither1)) << 16
            );
        StoreStreamed((unsigned short*)(exponents + h), (unsigned short)(next0 | next1 << 8));
    }
}

// 48 registers keep about 40 warps per SM resident without spilling; 40 registers spill and
// lose 5%. Items tile their pass.lanes, two opponent hands per thread, like Backup; a tile
// compacts its pairs carrying reach to its first threads, since zero-reach hands cluster and
// need only their children's mask bits.
__global__ void __launch_bounds__(kTileGroup, 5) Reach(GPU_ARGUMENTS)
{
    const State state = *statePointer;
    const U32 item = blockIdx.x, local = threadIdx.x, lane = blockIdx.y * blockDim.x + local;
    const Node n = LoadRecord(nodes + pass.offset + item);
    PrefetchNextNodes(nodes, pass, local);
    WaitForPredecessors();
    // Only the opponent's reach is propagated, one row of stride floats per slot: terminals
    // read it and the opponent's cumulative strategy accumulates it. The updating player's
    // own reach is never needed.
    const U32 p = 1 - state.player, count = p ? state.hands[1] : state.hands[0];
    // Two adjacent hands per thread.
    const U32 h = 2 * lane;
    const bool inside = h < count;
    const U32 pair = h + 1 < count ? 1u : 0u;
    // The lane's pair is bit warpLane of mask word `word` of a row (see MasksBuffer); the source
    // row's word loads beside the pair.
    const U32 maskWords = MaskWords(state.stride), word = lane / kWarpSize, warpLane = local % kWarpSize;
    const U32 sourceSlot = p ? n.reachSlot[1] : n.reachSlot[0];
    const float* source = scratch + U64(sourceSlot) * state.stride + h;
    const U32* sourceMask = masks + U64(sourceSlot) * maskWords + word;
    float value0 = 0.0f, value1 = 0.0f;
    if (inside)
    {
        if (IsRoot(n.info))
        {
            // The game root's reach is the hands' weights, zero for a padding hand.
            const float2 weight = *(const float2*)(weights + HandBase(state, p) + h);
            value0 = weight.x;
            value1 = weight.y;
        }
        else if (n.foldUtility[0] != 0.0f)
        {
            // Region roots and the children that derive their reach (see Node) apply the dealt
            // card and the outcome probability to the chance parent's reach. The pair's cards
            // share one word (see CardPairs); without a pair the second value is the zero
            // LoadPair supplies.
            const float2 entering = LoadMaskedPair(source, pair, *sourceMask >> warpLane);
            const U32 held = cards[CardPairs(state, p) + h / 2];
            value0 = HandBlocked(n.board, held) ? 0.0f : entering.x * n.foldUtility[0];
            value1 = HandBlocked(n.board, held >> 16) ? 0.0f : entering.y * n.foldUtility[0];
        }
        else
        {
            // The opponent's deeper decisions propagate its reach.
            const float2 entering = LoadMaskedPair(source, pair, *sourceMask >> warpLane);
            value0 = entering.x;
            value1 = entering.y;
        }
    }
    if (Kind(n.info) != NodeKind::Decision || Actor(n.info) != p)
    {
        // A region root other than the opponent's decision stores the opponent's reach entering
        // it for its descendants, every hand's, under an all-ones mask; an opponent decision's
        // children receive their own rows.
        if (inside)
            StorePair(scratch + U64(n.slot) * state.stride + h, value0, value1);
        if (warpLane == 0 && word < maskWords)
            masks[U64(n.slot) * maskWords + word] = 0xffffffffu;
        return;
    }
    const U32 actions = Count(n.info);
    // Pairs without reach leave their children's rows stale behind the rows' masks, each warp's
    // ballot stored to one child row per lane; the others move, in order, to the tile's first
    // threads: each warp counts its live pairs, which follow those of the warps before it.
    __shared__ U32 warpLive[kTileGroup / kWarpSize];
    __shared__ U32 livePairs[kTileGroup];
    __shared__ float2 liveValues[kTileGroup];
    const bool live = inside && (value0 != 0.0f || value1 != 0.0f);
    const U32 warp = local / kWarpSize, ballot = __ballot_sync(0xffffffffu, live);
    if (warpLane == 0)
        warpLive[warp] = __popc(ballot);
    static_assert(kMaxActions <= kWarpSize);
    if (warpLane < actions && word < maskWords)
        masks[U64(n.link + warpLane) * maskWords + word] = ballot;
    __syncthreads();
    // Lane w holds warp w's count.
    const U32 counted = warpLane < blockDim.x / kWarpSize ? warpLive[warpLane] : 0u;
    const U32 liveCount = __reduce_add_sync(0xffffffffu, counted), first = __reduce_add_sync(0xffffffffu, warpLane < warp ? counted : 0u);
    if (live)
    {
        const U32 position = first + __popc(ballot & ((1u << warpLane) - 1));
        livePairs[position] = h;
        liveValues[position] = make_float2(value0, value1);
    }
    __syncthreads();
    if (local >= liveCount)
        return;
    const U32 liveH = livePairs[local];
    const float2 v = liveValues[local];
    if (actions == 2)
        ReachNode<2>(n, state, p, liveH, v.x, v.y, regrets, sums, scratch);
    else if (actions == 3)
        ReachNode<3>(n, state, p, liveH, v.x, v.y, regrets, sums, scratch);
    else
        ReachNode<0>(n, state, p, liveH, v.x, v.y, regrets, sums, scratch);
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

// ScanSegments's path for segments of any length, over groups first to last of the lane's
// segment of length ranks: sums pass through shared memory, each rank's running sum from the
// first group to out, and the groups' total returns. Each kChunk groups load before any of their
// sums, so their latencies overlap; groups past the last add +0, as a group's ranks past the
// segment's end do.
__device__ __forceinline__ float ScanGroups(
    U32 first,
    U32 last,
    U32 length,
    const uint2* tokens,
    U32 lane,
    const unsigned char* staged,
    float* out
)
{
    constexpr U32 kChunk = 4;
    // Stores stop at the range's ranks: a chunk past last would store its unchanged sum into the
    // ranks of another warp's range.
    const U32 limit = 4 * last < length ? 4 * last : length;
    float sum = 0.0f;
    for (; first < last; first += kChunk)
    {
        float x[4 * kChunk];
#pragma unroll
        for (U32 g = 0; g < kChunk; ++g)
        {
            U32 offsets[4];
            Unpack(first + g < last ? tokens[(first + g) * kWarpSize + lane] : make_uint2(0u, 0u), offsets);
#pragma unroll
            for (U32 c = 0; c < 4; ++c)
                x[4 * g + c] = first + g < last ? Staged(staged, offsets[c]) : 0.0f;
        }
#pragma unroll
        for (U32 j = 0; j < 4 * kChunk; ++j)
        {
            sum += x[j];
            if (4 * first + j < limit)
                out[4 * first + j] = sum;
        }
    }
    return sum;
}

__device__ __forceinline__ void ScanSegmentsLong(U32 rankCount, const uint2* tokens, U32 lane, const unsigned char* staged, float* sums)
{
    const Segment bounds = LaneSegment(rankCount, lane);
    const U32 length = bounds.end - bounds.begin;
    float* out = sums + bounds.begin;
    const float carried = WarpBefore(ScanGroups(0, (length + 3) / 4, length, tokens, lane, staged, out), lane);
    for (U32 j = 0; j < length; ++j)
        out[j] += carried;
}

// A barrier of Threads threads, whole warps, on named barrier Barrier (0 is __syncthreads's): the
// warps sharing a doubled Terminal block's scan or its runs meet on one while the block's other
// warps go on.
template<U32 Barrier, U32 Threads>
__device__ __forceinline__ void NamedBarrier()
{
    static_assert(Barrier != 0 && Threads % kWarpSize == 0);
    __syncwarp();
    asm volatile("bar.sync %0, %1;" ::"n"(Barrier), "n"(Threads) : "memory");
}

// ScanSegmentsLong shared by two warps: each lane of warp half 0 scans the first half of its
// segment's groups and the same lane of half 1 the rest, each from zero. The halves' totals, their
// last running sums, pass through the prefix itself over the warps' barrier; then both add the
// segments before theirs and the second half its first half's total. The summation order differs
// from one warp's, as that differs from the CPU's sequential prefix.
__device__ __forceinline__ void ScanSegmentsSplit(
    U32 rankCount,
    const uint2* tokens,
    U32 lane,
    U32 half,
    const unsigned char* staged,
    float* sums
)
{
    const Segment bounds = LaneSegment(rankCount, lane);
    const U32 length = bounds.end - bounds.begin, groups = (length + 3) / 4, middle = (groups + 1) / 2;
    // The first half's ranks.
    const U32 split = 4 * middle < length ? 4 * middle : length;
    float* out = sums + bounds.begin;
    const float sum = ScanGroups(half ? middle : 0, half ? groups : middle, length, tokens, lane, staged, out);
    NamedBarrier<1, 2 * kWarpSize>();
    // An empty half's total is zero, not a stale slot's.
    const float first = half ? (split ? out[split - 1] : 0.0f) : sum;
    const float second = half ? sum : (length > split ? out[length - 1] : 0.0f);
    // Both halves have read the other's total before either adds its carry in place.
    NamedBarrier<1, 2 * kWarpSize>();
    const float carried = WarpBefore(first + second, lane) + (half ? first : 0.0f);
    for (U32 j = half ? split : 0, end = half ? length : split; j < end; ++j)
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

// Stages the opponent's reach row of slot row into reach, one float4 group per thread of the
// block's threads (rows and their staged copies are padded to whole groups, see TerminalLayout),
// and returns whether it carries reach. The row's mask words (see MasksBuffer) load beside its
// groups. A terminal without opponent reach leaves its values stale, and Backup substitutes zero
// from its flag.
__device__ __forceinline__ bool StageReach(
    State state,
    U32 row,
    U32 local,
    U32 threads,
    const float* scratch,
    const U32* masks,
    float* reach
)
{
    const U32 opponentCount = state.player ? state.hands[0] : state.hands[1];
    const float4* source = (const float4*)(scratch + U64(row) * state.stride);
    const U32* mask = masks + U64(row) * MaskWords(state.stride);
    bool nonzero = false;
    for (U32 group = local; 4 * group <= opponentCount; group += threads)
    {
        const float4 r = MaskedGroup(source, mask, group, opponentCount - 4 * group);
        ((float4*)reach)[group] = r;
        nonzero |= AnyReach(r);
    }
    return __syncthreads_or(nonzero);
}

// StageReach for a fold's own row, into foldReach, together with its showdown sibling's row,
// into reach, so their loads overlap; returns whether each carries reach: bit 0 the fold's,
// bit 1 the showdown's.
__device__ __forceinline__ U32 StageRows(
    State state,
    U32 foldRow,
    U32 showdownRow,
    U32 local,
    U32 threads,
    const float* scratch,
    const U32* masks,
    float* reach,
    float* foldReach
)
{
    const U32 opponentCount = state.player ? state.hands[0] : state.hands[1];
    const float4* foldSource = (const float4*)(scratch + U64(foldRow) * state.stride);
    const float4* source = (const float4*)(scratch + U64(showdownRow) * state.stride);
    const U32 maskWords = MaskWords(state.stride);
    const U32* foldMask = masks + U64(foldRow) * maskWords;
    const U32* mask = masks + U64(showdownRow) * maskWords;
    bool fold = false, showdown = false;
    for (U32 group = local; 4 * group <= opponentCount; group += threads)
    {
        const float4 f = MaskedGroup(foldSource, foldMask, group, opponentCount - 4 * group);
        const float4 r = MaskedGroup(source, mask, group, opponentCount - 4 * group);
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
// bounds the loop. A lane loads only its own run's groups and takes the zero slot's offsets (byte
// offset zero) for the others, so a group's loads touch only the sectors of the lanes it reaches.
// The run's total goes to totals[card] and, with Fold, the holders' mass in a fold sibling's own
// row, staged at foldStaged, to masses[1 + card], and the zero the hand loop reads for no holders
// before the run; a lane without a card stores none of them. With Split, two warps share the
// runs, half 0 taking each run's groups up to its middle one and half 1 those from it: the
// second half sums from zero until, over the warps' barrier, it adds the first half's total, the
// entry before its own, and completes the run's total and mass (a run of one group lies in the
// first half alone). groups then bounds the half's groups.
template<bool Fold, bool Split, U32 MaxGroups>
__device__ __forceinline__ void BlockerRun(
    U32 groups,
    U32 zero,
    U32 lane,
    U32 half,
    const uint2* tokens,
    const unsigned char* staged,
    const unsigned char* foldStaged,
    float* runs,
    float* masses,
    float* totals
)
{
    const U32 length = (lane >> 16) & 0xffu;
    // The lane's entries: count from the run's entry skip.
    U32 skip = 0, count = length;
    if constexpr (Split)
    {
        // The first half: the run's first (groups + 1) / 2 groups of four.
        const U32 middle = 4 * ((length + 7) / 8), first = middle < length ? middle : length;
        skip = half ? first : 0;
        count = half ? length - first : first;
    }
    float* run = (float*)((unsigned char*)runs + (lane & 0xffffu)) + skip;
    tokens += (skip / 4) * 2 * kWarpSize;
    float sum = 0.0f, foldMass = 0.0f;
    // Up to MaxGroups the groups' tokens load together before any sum; each group runs behind a
    // uniform test.
    const uint2 zeros = make_uint2(zero | zero << 16, zero | zero << 16);
    uint2 loaded[MaxGroups ? MaxGroups : 1];
    if constexpr (MaxGroups != 0)
    {
#pragma unroll
        for (U32 g = 0; g < MaxGroups; ++g)
            loaded[g] = count > 4 * g ? tokens[g * 2 * kWarpSize] : zeros;
    }
    // Holders past the run point at the staged rows' zero slot: their +0 leaves the sums.
#pragma unroll
    for (U32 g = 0; g < (MaxGroups ? MaxGroups : groups); ++g)
    {
        if (g >= groups)
            break;
        U32 offsets[4];
        Unpack(MaxGroups ? loaded[g] : count > 4 * g ? tokens[g * 2 * kWarpSize] : zeros, offsets);
#pragma unroll
        for (U32 c = 0; c < 4; ++c)
        {
            sum += Staged(staged, offsets[c]);
            if (Fold)
                foldMass += Staged(foldStaged, offsets[c]);
            if (count > 4 * g + c)
                run[4 * g + c] = sum;
        }
    }
    const U32 card = lane >> 24;
    if (card != 0xffu && (!Split || half == 0))
    {
        // Past the run, sum only added +0, so it is the run's last entry (+0 for no holders).
        totals[card] = sum;
        if (Fold)
            masses[1 + card] = foldMass;
        run[-1] = 0.0f;
    }
    if constexpr (Split)
    {
        NamedBarrier<2, 4 * kWarpSize>();
        // A lane without a card has no entries.
        if (half && count)
        {
            const float carried = run[-1];
            for (U32 j = 0; j < count; ++j)
                run[j] += carried;
            totals[card] = carried + sum;
            if (Fold)
                masses[1 + card] += foldMass;
        }
    }
}

template<bool Fold, bool Split>
__device__ __forceinline__ void BlockerRuns(
    U32 groups,
    U32 zero,
    U32 lane,
    U32 half,
    const uint2* tokens,
    const unsigned char* staged,
    const unsigned char* foldStaged,
    float* runs,
    float* masses,
    float* totals
)
{
    // A half's groups: the first (groups + 1) / 2 of the warp's, or the last groups / 2.
    if constexpr (Split)
        groups = half ? groups / 2 : (groups + 1) / 2;
    if (groups <= 6)
        BlockerRun<Fold, Split, 6>(groups, zero, lane, half, tokens, staged, foldStaged, runs, masses, totals);
    else
        BlockerRun<Fold, Split, 0>(groups, zero, lane, half, tokens, staged, foldStaged, runs, masses, totals);
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
// as the CPU evaluator's per-card masses. The first warp of the block's threads sums the total
// while the others take the card masses.
__device__ __forceinline__ void FoldMasses(State state, U32 local, U32 threads, const U32* cards, const float* reach, float* masses)
{
    const U32 opponent = 1 - state.player, opponentCount = state.player ? state.hands[0] : state.hands[1];
    if (local < kWarpSize)
        FoldTotal(opponentCount, local, reach, masses);
    else
        for (U32 card = local - kWarpSize; card < 52; card += threads - kWarpSize)
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
    U32 threads,
    const Hand* hands,
    const U32* cards,
    float* values,
    const float* reach,
    float* masses
)
{
    const U32 p = state.player;
    const U32 count = p ? state.hands[1] : state.hands[0], mineBase = HandBase(state, p);
    FoldMasses(state, local, threads, cards, reach, masses);
    __syncthreads();
    float* output = values + U64(n.slot) * state.stride;
    const float payoff = p ? n.utility[1][0] : n.utility[0][0];
    for (U32 h = local; h < count; h += threads)
    {
        // The hand: value scale and packed cards (see Hand).
        const uint2 hand = ((const uint2*)(hands + mineBase + h))[0];
        const float scale = BitsFloat(hand.x);
        const U32 identical = hand.y >> 16;
        output[h] = HandBlocked(n.board, hand.y) || scale <= 0.0f
                        ? 0.0f
                        : FoldValue(masses, hand.y, identical ? reach[identical - 1] : 0.0f, scale, payoff);
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
    const U32 stamp = LoadStreamed(stamps + read + index);
    // A live node's store does not wait for the load.
    if (writer && live)
        StoreStreamed(stamps + write + index, state.update);
    else if (writer)
        StoreStreamed(stamps + write + index, stamp);
    return Halving(state.update - 1u - stamp);
}

// One hand of a decision whose only children are the fold and showdown the calling block
// evaluated (see BackupLeafPair): its value and, when training, its regrets' new codes and
// exponent byte. r0 and r1 are its matched entries (see MatchedPair).
template<bool Averaging>
__device__ __forceinline__ float LeafHand(
    U32 actor,
    U32 ditherIndex,
    float r0,
    float r1,
    U32 exponent,
    State state,
    float child0,
    float child1,
    float halving,
    U32& code0,
    U32& code1,
    U32& next
)
{
    if (actor != state.player)
        return child0 + child1;
    if (!Averaging && state.evaluation != Evaluation::None)
        return child1 > child0 ? child1 : child0;
    const float positive = (r0 > 0.0f ? r0 : 0.0f) + (r1 > 0.0f ? r1 : 0.0f), reciprocal = RegretReciprocal(positive);
    const float value = child0 * MatchedProbability(r0, positive, reciprocal, 2) + child1 * MatchedProbability(r1, positive, reciprocal, 2);
    if (!Averaging)
    {
        const float updated0 = UpdatedRegret(Dequantize(r0, exponent), child0 - value, halving, state);
        const float updated1 = UpdatedRegret(Dequantize(r1, exponent), child1 - value, halving, state);
        next = RegretExponent(fmaxf(fabsf(updated0), fabsf(updated1)));
        const float dither = Dither(ditherIndex, state.update);
        code0 = U32(Quantize(updated0, next, dither)) & 0xffffu;
        code1 = U32(Quantize(updated1, next, dither)) & 0xffffu;
    }
    return value;
}

// Hands h and h + 1 (h even) of a decision whose only children are the fold and showdown the
// calling block evaluated: BackupNode's update without child loads. The decision has the given
// actor; entry0 and exponents point at hand h's first regret and exponent byte (its second regret
// is row units later), or with Averaging entry0 at its first cumulative strategy (see Terminal),
// ditherIndex is that regret's unpadded index, and values points at hand h's value in the
// decision's scratch row; action foldAction is the fold. The pair's entries share a word (see
// StateRow); past an odd hand count, hand h + 1 is padding, whose zero entries and values keep
// zero codes.
template<bool Averaging>
__device__ __forceinline__ void BackupLeafPair(
    U32 actor,
    U32 ditherIndex,
    short* entry0,
    unsigned char* exponents,
    U32 row,
    State state,
    U32 foldAction,
    float fold0,
    float showdown0,
    float fold1,
    float showdown1,
    float halving,
    float* values
)
{
    const bool regrets = actor == state.player && (Averaging || state.evaluation == Evaluation::None);
    U32 word0 = 0, word1 = 0, exponentPair = 0;
    if (regrets)
    {
        word0 = LoadStreamed((const U32*)entry0);
        word1 = LoadStreamed((const U32*)(entry0 + row));
        if (!Averaging)
            exponentPair = LoadStreamed((const unsigned short*)exponents);
    }
    const float2 entries0 = MatchedPair<Averaging>(word0), entries1 = MatchedPair<Averaging>(word1);
    U32 codes[2][2] = {}, next[2] = {};
    const float value0 = LeafHand<Averaging>(
        actor,
        ditherIndex,
        entries0.x,
        entries1.x,
        exponentPair & 0xffu,
        state,
        foldAction == 0 ? fold0 : showdown0,
        foldAction == 0 ? showdown0 : fold0,
        halving,
        codes[0][0],
        codes[0][1],
        next[0]
    );
    const float value1 = LeafHand<Averaging>(
        actor,
        ditherIndex + 1,
        entries0.y,
        entries1.y,
        exponentPair >> 8,
        state,
        foldAction == 0 ? fold1 : showdown1,
        foldAction == 0 ? showdown1 : fold1,
        halving,
        codes[1][0],
        codes[1][1],
        next[1]
    );
    if (!Averaging && regrets)
    {
        StoreStreamed((U32*)entry0, codes[0][0] | codes[1][0] << 16);
        StoreStreamed((U32*)(entry0 + row), codes[0][1] | codes[1][1] << 16);
        StoreStreamed((unsigned short*)exponents, (unsigned short)(next[0] | next[1] << 8));
    }
    StorePair(values, value0, value1);
}

// Showdowns (see Kernel), in blocks of Group threads, kTerminalGroup or, where shared memory
// limits resident blocks, twice that (see CudaExecutor's TerminalInstance). Registers are capped
// so 48 warps stay resident per SM. The first warp scans a showdown and the second sums a fold
// sibling's row while the next two build its blocker runs, one card per lane of its section's
// 2 * kWarpSize-entry lane table (see BuildSection in GpuPlan.cpp); all take part in the staging
// and the hand loop. A doubled block's warps 4, 6 and 7 take the second halves of the scan's
// segments and of the runs (see ScanSegmentsSplit and BlockerRun), meeting their first halves'
// warps on named barriers; its warp 5 idles until the hand loop.
static_assert(kTerminalGroup == 4 * kWarpSize);
// Averaging instances serve strategy-value evaluation (see Evaluation): acting parents back up by
// their average strategy. Separate instances keep that path out of the training kernels' code,
// where it cost wide ranges 16%.
template<bool Averaging, U32 Group>
__global__ void __launch_bounds__(Group, 12 * kTerminalGroup / Group) Terminal(GPU_ARGUMENTS)
{
    static_assert(Group == kTerminalGroup || Group == 2 * kTerminalGroup);
    const State state = *statePointer;
    const U32 item = blockIdx.x, local = threadIdx.x;
    const U32 p = state.player, opponent = 1 - p;
    // Fixed indices keep the small State/Node arrays in registers.
    const U32 count = p ? state.hands[1] : state.hands[0], opponentCount = p ? state.hands[0] : state.hands[1];
    // The grid has exactly one block per item (see CudaExecutor's Dispatch).
    const Node n = LoadRecord(nodes + pass.offset + item);
    // The shuffle tells the compiler that the warp index is uniform in the warp, so its branches
    // skip whole warps' work and keep shared addresses in uniform registers.
    const U32 warp = __shfl_sync(0xffffffffu, local / kWarpSize, 0), lane = local % kWarpSize;
    // Thread 0 moves the record 1024 items ahead into L2 for its block: a fixed distance tuned on
    // an 84-SM GPU, where twelve resident blocks per SM make it about one wave.
    if (warp == 0 && local == 0 && item + 1024 < pass.count)
        PrefetchL2(nodes + pass.offset + item + 1024);
    // Other warps have next-launch lines only when the pass has fewer than a warp's worth of
    // blocks per line (PrefetchNextNodes moves fewer than pass.nextCount + 2 lines).
    if (warp == 0 || kWarpSize * pass.count < pass.nextCount + 2)
        PrefetchNextNodes(nodes, pass, local);
    WaitForPredecessors();
    const U32 row = Row(n.info);
    const U32 mineBase = HandBase(state, p);
    const U32 reachSlot = p ? n.reachSlot[0] : n.reachSlot[1];
    // Rows of the opponent's hands (see TerminalLayout): the staged rows are padded to float4 groups.
    extern __shared__ __align__(16) float reach[];
    const TerminalLayout layout(opponentCount);
    float* prefix = reach + layout.forward;
    float* blockers = reach + layout.runs;
    const unsigned char* staged = (const unsigned char*)reach;
    // A fold's total and card masses, then a fold sibling's own row, past the blocker runs: a
    // live showdown sums the sibling's masses from that row while its scan and runs write the
    // prefix and run tables, and the hand loop reads all of them, the row after the fold's
    // values have replaced it in scratch.
    float* foldMasses = reach + layout.foldMasses;
    // Each card's run total (see BlockerRun), indexed by card.
    float* totals = reach + layout.totals;
    float* foldReach = reach + layout.foldReach;
    // A showdown evaluates its fold sibling from the same opponent reach row when the
    // updating player folds there, otherwise from the masses of the fold's own row, and backs
    // the parent up when the two are its only children (see Node).
    const bool sibling = n.fold != kNoIndex;
    // A showdown with a fold sibling carries the parent fields the block reads (see Node).
    const bool ownFold = sibling && Actor(n.info) == p;
    const bool backup = sibling && Count(n.info) == 2;
    // As in BackupNode, an acting parent's regrets train and its stamp advances at every training
    // update.
    const bool trainParent = backup && ownFold && state.evaluation == Evaluation::None;
    // Siblings hold consecutive slots, so the fold is action 0 when its slot comes first.
    const U32 foldAction = n.fold < n.slot ? 0u : 1u;
    bool liveFold, live;
    if (sibling && !ownFold)
    {
        // The fold's own row, propagated by Reach, stages with the showdown's row; the hand
        // loop below takes the fold's values from its masses, which a live showdown's second
        // warp and blocker runs sum along.
        const U32 any = StageRows(state, n.fold, reachSlot, local, Group, scratch, masks, reach, foldReach);
        liveFold = (any & 1u) != 0;
        live = (any & 2u) != 0;
        if (liveFold && !live)
            FoldMasses(state, local, Group, cards, foldReach, foldMasses);
    }
    else
    {
        live = StageReach(state, reachSlot, local, Group, scratch, masks, reach);
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
    // contiguous, 4 * row + count bytes: one thread per 128-byte line, the last covering the final byte.
    const U32 legal = (n.rankCounts >> (16 * opponent)) & 0xffffu;
    const U32 parentRow = StateRow(count), parentBytes = 4 * parentRow + count;
    // Warps past the parent's lines skip the prefetch's address arithmetic.
    if (trainParent && 128 * kWarpSize * warp < parentBytes + 127 && 128 * local < parentBytes + 127)
        PrefetchL2((const unsigned char*)(regrets + StateOffset(n.strategy)) + (128 * local < parentBytes ? 128 * local : parentBytes - 1));
    if (live)
    {
        // The row's section for the opponent's hands (see Plan::order): a header, the scan
        // tokens, the lane table and the run tokens.
        const uint4* section =
            (const uint4*)(order + U64(row) * state.orderPitch + (opponent ? state.orderSection : OrderEntryWords(state)));
        // The first warp scans the reach in rank order; the second sums a fold sibling's own row.
        constexpr bool split = Group != kTerminalGroup;
        const U32 role = split ? warp % (kTerminalGroup / kWarpSize) : warp, half = split ? warp / (kTerminalGroup / kWarpSize) : 0u;
        if (role == 0)
        {
            if (split)
                ScanSegmentsSplit(legal, (const uint2*)(section + 1), lane, half, staged, prefix);
            else
                TerminalScan(legal, (const uint2*)(section + 1), lane, staged, prefix);
        }
        else if (role == 1)
        {
            if (half == 0 && foldFromRow)
                FoldTotal(opponentCount, lane, foldReach, foldMasses);
        }
        else
        {
            // Showdowns accumulate each card's blocked reach in rank order once per node, as the
            // CPU evaluator's per-card masses do, instead of rescanning blockers per hand. The
            // third and fourth warps build them while the first scans, one card per lane, together
            // with a fold sibling's card masses from its own row.
            // The lane table's place follows from the rank count, so its entries load with the
            // header; a lane's run length predicates its token loads.
            const uint4 header = section[0];
            const U32 runLane = local % kTerminalGroup - 2 * kWarpSize;
            const uint4* table = section + SectionLaneTable(legal);
            const U32 laneRun = ((const U32*)table)[runLane];
            const U32 groups = role == 2 ? header.x & 0xffu : (header.x >> 8) & 0xffu;
            const uint2* tokens = (const uint2*)(table + 16) + runLane;
            const U32 zero = U32(sizeof(float)) * opponentCount;
            if (foldFromRow)
                BlockerRuns<true, split>(
                    groups, zero, laneRun, half, tokens, staged, (const unsigned char*)foldReach, blockers, foldMasses, totals
                );
            else
                BlockerRuns<false, split>(groups, zero, laneRun, half, tokens, staged, staged, blockers, foldMasses, totals);
        }
    }
    __syncthreads();
    const float win = p ? n.utility[1][0] : n.utility[0][0];
    const float tie = p ? n.utility[1][1] : n.utility[0][1];
    const float loss = p ? n.utility[1][2] : n.utility[0][2];
    const float foldPayoff = p ? n.foldUtility[1] : n.foldUtility[0];
    // As in the CPU evaluator, zero-tie showdowns need only win/loss masses; a fold from
    // the same row needs the mass. Loss masses are the ranked mass less the forward prefix.
    const bool needMass = tie != 0.0f || ownFold;
    const float rankedMass = live && legal ? prefix[legal - 1] : 0.0f;
    const float winScale = win - tie, lossScale = loss - tie;
    // The loop indexes block-uniform bases by the hand: per-thread pointers advancing with it
    // would hold registers that its loads need.
    const Hand* mine = hands + mineBase;
    const uint4* rowEntries = (const uint4*)(order + U64(row) * state.orderPitch) + mineBase / 2;
    // Values replace reach rows in scratch once the block has staged the reach it reads, a
    // backed-up parent's replacing the showdown's.
    float* output = scratch + U64(backup ? n.link : n.slot) * state.stride;
    float* foldOutput = scratch + U64(sibling ? n.fold : n.slot) * state.stride;
    // Averaging parents read their cumulative strategies in the regrets' place.
    short* parentRegrets = (Averaging ? (short*)sums : regrets) + StateOffset(n.strategy);
    unsigned char* parentExponents = ExponentBytes(parentRegrets, 2, parentRow);
    // One hand's showdown and fold values from its record and order entries.
    const auto evaluate = [&](uint2 first, uint2 entries, float& showdown, float& fold)
    {
        const float scale = BitsFloat(first.x);
        const U32 identical = first.y >> 16;
        showdown = 0.0f;
        fold = 0.0f;
        const bool open = entries.x != kNoIndex;
        if (live && open)
        {
            float mass = 0.0f;
            // Where the hand's masses are, which depends only on the board and ranges, in the
            // staged reach (see Plan::order): its win mass, the ranked mass not stronger than it
            // in the forward prefix, and per held card its blocked reach through its rank in the
            // card's run and, the given count of entries earlier, below it, as byte offsets; the
            // runs' totals are in totals.
            const float winMass = Staged(staged, entries.x & 0x3ffcu);
            const float lossMass = rankedMass - Staged(staged, (entries.x >> 12) & 0x3ffcu);
            const float all0 = totals[first.y & 0xffu], all1 = totals[(first.y >> 8) & 0xffu];
            const U32 at0 = (entries.y << 2) & 0x7ffcu, at1 = (entries.y >> 11) & 0x7ffcu;
            const float below0 = Staged(staged, at0 - 4 * (entries.x >> 26)), through0 = Staged(staged, at0);
            const float below1 = Staged(staged, at1 - 4 * (entries.y >> 26)), through1 = Staged(staged, at1);
            if (needMass)
            {
                mass = rankedMass - (all0 + all1);
                if (identical)
                    mass += reach[identical - 1];
            }
            const float wins = winMass - (below0 + below1);
            // A card's list has at most 51 entries, so all - through loses only ulps of that
            // card's mass.
            const float losses = lossMass - ((all0 - through0) + (all1 - through1));
            showdown = (tie * mass + winScale * wins + lossScale * losses) * scale;
            if (ownFold)
                fold = (mass * scale) * foldPayoff;
        }
        if (foldFromRow && open)
            fold = FoldValue(foldMasses, first.y, identical ? foldReach[identical - 1] : 0.0f, scale, foldPayoff);
    };
    // A thread takes hands h and h + 1, whose parent entries share words (see BackupLeafPair);
    // past an odd count, hand h + 1 is padding, which is closed (see Plan::order).
    const U32 half = (count + 1) / 2;
    for (U32 k = local; k < half; k += Group)
    {
        const U32 h = 2 * k;
        // The hands' records: value scales and packed cards (see Hand and HandBase), and their
        // order entries, kNoIndex words for a hand the row's board blocks or without a value scale
        // (see Plan::order).
        const uint4 records = *(const uint4*)(mine + h), entries = rowEntries[k];
        float showdown0, fold0, showdown1, fold1;
        evaluate(make_uint2(records.x, records.y), make_uint2(entries.x, entries.y), showdown0, fold0);
        evaluate(make_uint2(records.z, records.w), make_uint2(entries.z, entries.w), showdown1, fold1);
        if (!backup)
        {
            if (live)
                StorePair(output + h, showdown0, showdown1);
            if (liveFold)
                StorePair(foldOutput + h, fold0, fold1);
        }
        else
            BackupLeafPair<Averaging>(
                Actor(n.info),
                U32(n.strategy) + h,
                parentRegrets + h,
                parentExponents + h,
                parentRow,
                state,
                foldAction,
                fold0,
                showdown0,
                fold1,
                showdown1,
                halving,
                output + h
            );
    }
}

// The 16-bit entries of runout row row > 0 in its player-0-major (major 0) or player-1-major
// layout, after the flop row's two 32-bit layouts (see Outcomes); row 0 gives row 1's.
__device__ __forceinline__ unsigned short* TurnOutcomes(U32* outcomes, U64 pairs, U32 row, U32 major)
{
    return (unsigned short*)(outcomes + 2 * pairs) + (2 * U64(row ? row - 1 : 0) + major) * pairs;
}

// Two adjacent hands' runout values, h's and, when second, h + 1's, from their row of win/loss
// counts against the opponent hands (see Outcomes) with reach: offsets holds their entries' offsets
// in the row and kept their reach, in groups of eight padded with zero reach (see Runout). Entry is
// the row's entry type, whose low half holds player 0's wins and high half its losses, and p the
// updating player, whose wins and losses the two halves' sums become. This player's hands are
// contiguous per opponent hand in its layout of the row, so a warp's loads coalesce and the two
// hands' entries share an address; a missing second hand reads as the sentinel. A group's loads
// are independent and overlap the row's latency while the sums keep their opponent order over the
// opponent hands with reach. Overlapping pairs hold the sentinel and are skipped, as are the
// padding's terms, exact zeros. A runout record's win and loss are the scales of its counts (see
// Node).
template<typename Entry>
__device__ __forceinline__ void RunoutValues(
    const Entry* table,
    U32 p,
    bool second,
    U32 groups,
    const U32* offsets,
    const float* kept,
    float win,
    float tie,
    float loss,
    float scale0,
    float scale1,
    float& value0,
    float& value1
)
{
    constexpr U32 kHalf = 4 * sizeof(Entry), kSentinel = Entry(~0u), kLow = (1u << kHalf) - 1;
    float mass0 = 0.0f, lows0 = 0.0f, highs0 = 0.0f, mass1 = 0.0f, lows1 = 0.0f, highs1 = 0.0f;
    // An opaque base keeps each entry's address one wide multiply-add from it.
    asm("" : "+l"(table));
#define GPU_RUNOUT_STEP(counts, r, mass, lows, highs) \
    if ((counts) != kSentinel)                        \
    {                                                 \
        mass += (r);                                  \
        lows += (r) * float((counts) & kLow);         \
        highs += (r) * float((counts) >> kHalf);      \
    }
    for (U32 g = 0; g < groups; ++g)
    {
        const uint4 o0 = ((const uint4*)offsets)[2 * g], o1 = ((const uint4*)offsets)[2 * g + 1];
        const U32 o[8] = {o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w};
        U32 a[8], b[8];
#pragma unroll
        for (U32 i = 0; i < 8; ++i)
        {
            const Entry* entry = table + o[i];
            a[i] = __ldg(entry);
            b[i] = second ? U32(__ldg(entry + 1)) : kSentinel;
        }
        const float4 r0 = ((const float4*)kept)[2 * g], r1 = ((const float4*)kept)[2 * g + 1];
        const float r[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
        for (U32 i = 0; i < 8; ++i)
        {
            GPU_RUNOUT_STEP(a[i], r[i], mass0, lows0, highs0)
            GPU_RUNOUT_STEP(b[i], r[i], mass1, lows1, highs1)
        }
    }
#undef GPU_RUNOUT_STEP
    const float wins0 = p ? highs0 : lows0, losses0 = p ? lows0 : highs0;
    const float wins1 = p ? highs1 : lows1, losses1 = p ? lows1 : highs1;
    value0 = (tie * mass0 + wins0 * win + losses0 * loss) * scale0;
    value1 = (tie * mass1 + wins1 * win + losses1 * loss) * scale1;
}

// Lone folds and forced runouts (see Kernel): one block per item and one thread per two adjacent
// hands (see RunoutValues), all in one round (see CudaExecutor's RunoutGroup), since a runout's
// hand visits every opponent hand with reach. Shared memory holds the staged opponent reach, then a fold's masses or a
// runout's reached opponent hands and their reach (see RunoutShared).
__global__ void __launch_bounds__(kRunoutGroup) Runout(GPU_ARGUMENTS)
{
    const State state = *statePointer;
    const U32 local = threadIdx.x, threads = blockDim.x;
    const U32 p = state.player;
    const U32 count = p ? state.hands[1] : state.hands[0], opponentCount = p ? state.hands[0] : state.hands[1];
    const Node n = LoadRecord(nodes + pass.offset + blockIdx.x);
    PrefetchNextNodes(nodes, pass, local);
    WaitForPredecessors();
    extern __shared__ __align__(16) float reach[];
    float* masses = reach + TerminalLayout::Padded(opponentCount + 1);
    const bool live = StageReach(state, p ? n.reachSlot[0] : n.reachSlot[1], local, threads, scratch, masks, reach);
    if (local == 0)
        flags[n.slot] = live ? 1u : 0u;
    if (!live)
        return;
    if (Kind(n.info) == NodeKind::Fold)
    {
        TerminalFold(n, state, local, threads, hands, cards, scratch, reach, masses);
        return;
    }
    // A runout's hands visit only the opponent hands with reach, listed in place of a fold's
    // masses (a zero-reach hand's terms would add exact zeros), then replaced by their entries'
    // offsets in the row, with their reach compacted beside, both padded with zero reach to
    // whole groups of eight.
    U32* kept = (U32*)masses;
    float* keptReach = masses + RunoutKept(opponentCount);
    __shared__ U32 keptCount;
    if (local < kWarpSize)
        ListReached(opponentCount, local, reach, kept, &keptCount);
    __syncthreads();
    const U32 reached = keptCount, padded = RunoutKept(reached);
    for (U32 k = local; k < padded; k += threads)
    {
        const U32 other = k < reached ? kept[k] : 0u;
        keptReach[k] = k < reached ? reach[other] : 0.0f;
        kept[k] = other * count;
    }
    __syncthreads();
    const float win = p ? n.utility[1][0] : n.utility[0][0];
    const float tie = p ? n.utility[1][1] : n.utility[0][1];
    const float loss = p ? n.utility[1][2] : n.utility[0][2];
    const U32 mineBase = HandBase(state, p), groups = padded / 8, row = Row(n.info);
    // Player 0 reads the player-1-major layout of the row, where its hands are contiguous.
    const U64 pairs = U64(state.hands[0]) * state.hands[1];
    const U32* flopTable = outcomes + (p == 0 ? pairs : 0);
    const unsigned short* turnTable = TurnOutcomes(outcomes, pairs, row, p == 0 ? 1 : 0);
    float* output = scratch + U64(n.slot) * state.stride;
    // Each thread takes two adjacent hands (see RunoutValues), in one round.
    for (U32 h = 2 * local; h < count; h += 2 * threads)
    {
        // The hands' records: value scales and packed cards (see Hand and HandBase); past an odd
        // count, hand h + 1 is padding, whose zero scale is never live.
        const bool second = h + 1 < count;
        const uint4 records = *(const uint4*)(hands + mineBase + h);
        const float scale0 = BitsFloat(records.x), scale1 = BitsFloat(records.z);
        const bool live0 = !HandBlocked(n.board, records.y) && scale0 > 0.0f;
        const bool live1 = !HandBlocked(n.board, records.w) && scale1 > 0.0f;
        float value0 = 0.0f, value1 = 0.0f;
        if (live0 || live1)
        {
            if (row == 0)
                RunoutValues(flopTable + h, p, second, groups, kept, keptReach, win, tie, loss, scale0, scale1, value0, value1);
            else
                RunoutValues(turnTable + h, p, second, groups, kept, keptReach, win, tie, loss, scale0, scale1, value0, value1);
        }
        StorePair(output + h, live0 ? value0 : 0.0f, live1 ? value1 : 0.0f);
    }
}

// Small decisions retain entry data in registers and issue every load before the first store
// so latencies overlap; a zero ActionCount keeps the general path, whose entry arrays live in
// local memory and whose loops stay rolled (see ReachNode). A thread carries two adjacent
// hands so twice the value loads are in flight per warp, and their entries share words (see
// StateRow); pair is zero when only hand h exists, whose second hand's zero values (see
// LoadPair) and padding entries update to zero. Child values load with their flags, before
// the node's liveness is known, and dead children's are replaced by zero, so the flag and
// value latencies overlap; regrets wait for liveness, which spares pruned nodes their DRAM
// traffic. A node is live when any child subtree carries opponent reach; pruned nodes have
// zero values and keep their regrets, whose skipped discounts the stamps apply at the next
// live update. An update re-encodes each hand's regrets (see GpuQuantize.h). With Averaging,
// regrets points at the cumulative strategies instead, whose average strategy weights an
// acting decision's children (see Terminal).
template<U32 ActionCount, bool Averaging>
__device__ __forceinline__ void BackupNode(Node n, State state, U32 h, U32 pair, short* regrets, float* values, U32* flags, U32* stamps)
{
    const U32 count = state.player ? state.hands[1] : state.hands[0];
    const U32 actions = ActionCount ? ActionCount : Count(n.info);
    const bool acting = Kind(n.info) == NodeKind::Decision && Actor(n.info) == state.player;
    const bool training = !Averaging && acting && state.evaluation == Evaluation::None;
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
            const float* row = values + U64(first) * state.stride + h;
            for (U32 k = 0; k < 3; ++k)
            {
                const bool present = k == 0 || k < sources;
                const float2 source = present ? LoadPair(row + k * state.stride, pair) : make_float2(0.0f, 0.0f);
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
    {
        const float* row = values + U64(n.link) * state.stride + h;
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            const U32 flag = flags[n.link + a];
            lives[a] = flag;
            live |= flag;
            const float2 source = LoadPair(row + a * state.stride, pair);
            childValues0[a] = source.x;
            childValues1[a] = source.y;
        }
    }
    if (h == 0)
        flags[n.slot] = live;
    const float halving = training ? AdvanceStamp(state, n.stamp, live != 0, h == 0, stamps) : 1.0f;
    float* output = values + U64(n.slot) * state.stride + h;
    if (!live)
    {
        if (IsRoot(n.info))
            StorePair(output, 0.0f, 0.0f);
        return;
    }
    // Reuse the entry-policy denominator for value backup and the regret update.
    float positive0 = 0.0f, positive1 = 0.0f;
    // The pair's entries share a word (see StateRow).
    const U32 row = StateRow(count);
    short* units = regrets + StateOffset(n.strategy) + h;
    unsigned char* exponents = ExponentBytes(regrets + StateOffset(n.strategy), actions, row);
    U32 exponent0 = 0, exponent1 = 0;
    if (Averaging ? acting : training)
    {
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            const float2 r = MatchedPair<Averaging>(LoadStreamed((const U32*)(units + U64(a) * row)));
            entryRegrets0[a] = r.x;
            entryRegrets1[a] = r.y;
            positive0 += r.x > 0.0f ? r.x : 0.0f;
            positive1 += r.y > 0.0f ? r.y : 0.0f;
        }
        if (training)
        {
            const U32 exponentPair = LoadStreamed((const unsigned short*)(exponents + h));
            exponent0 = exponentPair & 0xffu;
            exponent1 = exponentPair >> 8;
        }
    }
    // Dead children's values are zero. Each mode has a loop of its own, which keeps the unrolled
    // loops free of branches.
#pragma unroll(ActionCount ? ActionCount : 1)
    for (U32 a = 0; a < actions; ++a)
        if (!lives[a])
            childValues0[a] = childValues1[a] = 0.0f;
    float value0 = 0.0f, value1 = 0.0f;
    if (acting && !Averaging && state.evaluation != Evaluation::None)
    {
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            value0 = a == 0 || childValues0[a] > value0 ? childValues0[a] : value0;
            value1 = a == 0 || childValues1[a] > value1 ? childValues1[a] : value1;
        }
    }
    else if (acting)
    {
        const float reciprocal0 = RegretReciprocal(positive0), reciprocal1 = RegretReciprocal(positive1);
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            value0 += childValues0[a] * MatchedProbability(entryRegrets0[a], positive0, reciprocal0, actions);
            value1 += childValues1[a] * MatchedProbability(entryRegrets1[a], positive1, reciprocal1, actions);
        }
    }
    else
    {
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            value0 += childValues0[a];
            value1 += childValues1[a];
        }
    }
    StorePair(output, value0, value1);
    if (training)
    {
        float magnitude0 = 0.0f, magnitude1 = 0.0f;
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            entryRegrets0[a] = UpdatedRegret(Dequantize(entryRegrets0[a], exponent0), childValues0[a] - value0, halving, state);
            entryRegrets1[a] = UpdatedRegret(Dequantize(entryRegrets1[a], exponent1), childValues1[a] - value1, halving, state);
            magnitude0 = fmaxf(magnitude0, fabsf(entryRegrets0[a]));
            magnitude1 = fmaxf(magnitude1, fabsf(entryRegrets1[a]));
        }
        const U32 next0 = RegretExponent(magnitude0), next1 = RegretExponent(magnitude1);
        const float dither0 = Dither(U32(n.strategy) + h, state.update), dither1 = Dither(U32(n.strategy) + h + 1, state.update);
#pragma unroll(ActionCount ? ActionCount : 1)
        for (U32 a = 0; a < actions; ++a)
        {
            const U32 low = U32(Quantize(entryRegrets0[a], next0, dither0)) & 0xffffu;
            const U32 high = U32(Quantize(entryRegrets1[a], next1, dither1)) << 16;
            StoreStreamed((U32*)(units + U64(a) * row), low | high);
        }
        StoreStreamed((unsigned short*)(exponents + h), (unsigned short)(next0 | next1 << 8));
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
    const float* row = values + U64(n.link) * state.stride + h;
    // The children's rows and flags were last written when their subtrees finished, mostly long
    // enough ago to have left the L2: all move into it at once, so each batch below waits for the
    // L2 instead of a DRAM round of its own.
#pragma unroll 1
    for (U32 a = 0; a < children; ++a)
        PrefetchL2(row + U64(a) * state.stride);
    if (h == 0)
    {
        PrefetchL2(flags + n.link);
        PrefetchL2(flags + n.link + children / 2);
        PrefetchL2(flags + n.link + children - 1);
    }
#pragma unroll 1
    for (U32 a = 0; a < children; a += 8, row += 8 * state.stride)
    {
        U32 flag[8];
        float loaded0[8], loaded1[8];
        for (U32 k = 0; k < 8; ++k)
        {
            const bool present = a + k < children;
            const float2 source = present ? LoadPair(row + k * state.stride, pair) : make_float2(0.0f, 0.0f);
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
    StorePair(output, value0, value1);
}

// 48 registers keep 42 warps per SM resident for both hand-tile sizes instead of 36. Averaging
// instances serve strategy-value evaluation (see Terminal).
template<bool Averaging>
__global__ void __launch_bounds__(kTileGroup, 5) Backup(GPU_ARGUMENTS)
{
    const State state = *statePointer;
    const U32 item = blockIdx.x, h = 2 * (blockIdx.y * blockDim.x + threadIdx.x);
    const U32 count = state.player ? state.hands[1] : state.hands[0];
    PrefetchNextNodes(nodes, pass, threadIdx.x);
    // The grid has one block row per item (see CudaExecutor's Dispatch); a last hand tile can
    // run past the hands.
    if (h >= count)
        return;
    const U32 pair = h + 1 < count ? 1u : 0u;
    const Node n = LoadRecord(nodes + pass.offset + item);
    WaitForPredecessors();
    const U32 actions = Count(n.info);
    short* entries = Averaging ? (short*)sums : regrets;
    if (Kind(n.info) == NodeKind::Chance)
        BackupChance(n, state, h, pair, scratch, flags);
    else if (actions == 2)
        BackupNode<2, Averaging>(n, state, h, pair, entries, scratch, flags, stamps);
    else if (actions == 3)
        BackupNode<3, Averaging>(n, state, h, pair, entries, scratch, flags, stamps);
    else
        BackupNode<0, Averaging>(n, state, h, pair, entries, scratch, flags, stamps);
}

__global__ void Outcomes(GPU_ARGUMENTS)
{
    const State state = *statePointer;
    const U64 index = blockIdx.x * blockDim.x + threadIdx.x, pairs = U64(state.hands[0]) * state.hands[1];
    if (index >= pass.count)
        return;
    const U64 pair = index % pairs;
    const U32 row = U32(index / pairs), h0 = U32(pair / state.hands[1]), h1 = U32(index % state.hands[1]);
    const U64 mask0 = HandMask(hands[h0].cards), mask1 = HandMask(hands[HandBase(state, 1) + h1].cards);
    const U64 board = state.board;
    // Each row's player-1-major copy follows its player-0-major entries: the flop row's 32-bit
    // entries hold wins | losses << 16 of its 990 runouts, the turn rows' 16-bit entries that
    // follow wins | losses << 8 of their 44.
    const U64 transposed = pairs + U64(h1) * state.hands[0] + h0;
    unsigned short* turns = TurnOutcomes(outcomes, pairs, row, 0);
    const auto store = [&](U32 wins, U32 losses)
    {
        if (row == 0)
            outcomes[pair] = outcomes[transposed] = wins | losses << 16;
        else
            turns[pair] = turns[transposed] = (unsigned short)(wins | losses << 8);
    };
    U32 wins = 0, losses = 0;
    // Overlapping private cards store a sentinel, all ones, that Runout's loop tests instead of
    // masks.
    if (mask0 & mask1)
    {
        store(0xffffu, 0xffffu);
        return;
    }
    if (pass.outcomeStage == OutcomeStage::SumTurns)
    {
        for (U32 r = 1; r < state.outcomeRows; ++r)
        {
            const U32 result = TurnOutcomes(outcomes, pairs, r, 0)[pair];
            wins += result & 0xffu;
            losses += result >> 8;
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
                const U64 offset = U64(rankRow) * PaddedHands(state);
                const U32 a = ranks[offset + h0], b = ranks[offset + HandBase(state, 1) + h1];
                wins += a > b;
                losses += a < b;
            }
        }
    }
    store(wins, losses);
}
