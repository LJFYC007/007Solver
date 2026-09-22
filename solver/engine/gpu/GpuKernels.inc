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
#define GPU_WIDTH 32u
#define GPU_SHARED
#define GPU_GROUP_TAIL
#define GPU_GROUP_MEMORY extern __shared__ float groupMemory[];
#define GPU_GROUP_INDEX blockIdx.x
#define GPU_LOCAL_INDEX threadIdx.x
#define GPU_GROUP_SIZE blockDim.x
#define GPU_GROUP_BARRIER() __syncthreads()
#endif
#define GPU_ARGUMENTS                                                                                                                  \
    GPU_BUFFER(const Node, nodes, NodesBuffer), GPU_BUFFER(const U32, childSlots, ChildSlotsBuffer),                                   \
        GPU_BUFFER(const Hand, hands, HandsBuffer), GPU_BUFFER(const unsigned short, ranks, RanksBuffer),                              \
        GPU_BUFFER(const int, runouts, RunoutsBuffer), GPU_BUFFER(const U32, order, OrderBuffer),                                      \
        GPU_BUFFER(const U32, cards, CardsBuffer), GPU_BUFFER(const U32, work, WorkBuffer), GPU_BUFFER(U32, outcomes, OutcomesBuffer), \
        GPU_BUFFER(float, regrets, RegretsBuffer), GPU_BUFFER(float, sums, SumsBuffer), GPU_BUFFER(float, scratch, ScratchBuffer),     \
        GPU_BUFFER(float, values, ValuesBuffer), GPU_TAIL
#ifdef __METAL_VERSION__
#define GPU_STATE
#else
#define GPU_STATE const State state = *statePointer;
#endif

GPU_INLINE float Policy(
    Node n,
    U32 hand,
    U32 action,
    U32 count,
    bool average,
    GPU_DEVICE const float* regrets,
    GPU_DEVICE const float* sums
)
{
    float total = 0.0f;
    for (U32 a = 0; a < n.count; ++a)
    {
        const float x = average ? sums[n.strategy + U64(a) * count + hand] : regrets[n.strategy + U64(a) * count + hand];
        total += x > 0.0f ? x : 0.0f;
    }
    const float x = average ? sums[n.strategy + U64(action) * count + hand] : regrets[n.strategy + U64(action) * count + hand];
    return total > 0.0f ? (x > 0.0f ? x : 0.0f) / total : 1.0f / float(n.count);
}

GPU_ENTRY(Reach)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U32 index = GPU_INDEX;
    const U32 item = index / state.stride, h = index % state.stride;
    if (item >= pass.count)
        return;
    const Node n = nodes[work[pass.offset + item]];
    const U32 total = state.hands[0] + state.hands[1];
    for (U32 p = 0; p < 2; ++p)
    {
        if (h >= state.hands[p])
            continue;
        // Own reach is consumed only by training updates at nonterminal nodes.
        if (p == state.player && (state.evaluation || (n.kind != NodeKind::Decision && n.kind != NodeKind::Chance)))
            continue;
        const U32 base = p ? state.hands[0] : 0;
        const Hand hand = hands[base + h];
        float value;
        if (n.parent == 0xffffffffu)
            value = p == state.player ? 1.0f : hand.weight;
        else
        {
            const Node parent = nodes[n.parent];
            value = scratch[U64(parent.slot) * total + base + h];
            if (parent.kind == NodeKind::Chance)
                value = hand.mask & (U64(1) << n.dealtCard) ? 0.0f : value * (p == state.player ? 1.0f : 1.0f / float(parent.count - 4));
            else if (parent.actor == p)
                value *= Policy(parent, h, n.action, state.hands[p], state.evaluation != 0, regrets, sums);
        }
        scratch[U64(n.slot) * total + base + h] = value;
    }
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
    GPU_DEVICE const U32* sorted = order + U64(n.rankRow) * 2 * total + base;
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
    const Node n = nodes[work[pass.offset + item]];
    const U32 total = state.hands[0] + state.hands[1], mineBase = p ? state.hands[0] : 0, oppBase = p ? 0 : state.hands[0];
    GPU_SHARED float* reach = groupMemory;
    GPU_SHARED float* prefix = groupMemory + state.stride;
    for (U32 other = local; other < opponentCount; other += groupWidth)
        reach[other] = scratch[U64(n.slot) * total + oppBase + other];
    GPU_GROUP_BARRIER();
    for (U32 direction = local / GPU_WIDTH; direction < 2; direction += groupWidth / GPU_WIDTH)
        TerminalPrefix(n, state, local % GPU_WIDTH, GPU_WIDTH, direction != 0, order, reach, prefix);
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
        const float win = p == 0 ? n.utility[0] : -n.utility[2];
        const float tie = p == 0 ? n.utility[1] : -n.utility[1];
        const float loss = p == 0 ? n.utility[2] : -n.utility[0];
        float mass = 0.0f, wins = 0.0f, losses = 0.0f;
        if (n.kind == NodeKind::ForcedRunout)
        {
            for (U32 other = 0; other < opponentCount; ++other)
            {
                if (mine.mask & hands[oppBase + other].mask)
                    continue;
                const U64 pair = p == 0 ? U64(h) * state.hands[1] + other : U64(other) * state.hands[1] + h;
                const U32 counts = outcomes[U64(n.outcomeRow) * state.hands[0] * state.hands[1] + pair];
                mass += reach[other];
                wins += reach[other] * float(p == 0 ? counts & 0xffffu : counts >> 16);
                losses += reach[other] * float(p == 0 ? counts >> 16 : counts & 0xffffu);
            }
            const float divisor = n.outcomeRow == 0 ? 990.0f : 44.0f;
            output[h] = (tie * mass + (win - tie) * (wins / divisor) + (loss - tie) * (losses / divisor)) / mine.divisor;
            continue;
        }
        const U32 rank = n.kind == NodeKind::Showdown ? ranks[U64(n.rankRow) * total + mineBase + h] : 0;
        const U32 legal = n.kind == NodeKind::Showdown ? (p ? n.rankCounts[0] : n.rankCounts[1]) : opponentCount;
        if (n.kind == NodeKind::Showdown)
        {
            // Rank boundaries depend only on the board and ranges, not terminal reach.
            const U32 bounds = order[U64(n.rankRow) * 2 * total + total + mineBase + h];
            const U32 lower = bounds & 0xffffu, upper = bounds >> 16;
            wins = lower ? prefix[lower - 1] : 0.0f;
            losses = upper < legal ? prefix[state.stride + upper] : 0.0f;
        }
        // As in the CPU evaluator, zero-tie showdowns need only win/loss masses.
        const bool needMass = n.kind == NodeKind::Fold || tie != 0.0f;
        mass = needMass ? (n.kind == NodeKind::Fold ? prefix[0] : legal ? prefix[legal - 1] : 0.0f) : 0.0f;
        float blocked = 0.0f, blockedWins = 0.0f, blockedLosses = 0.0f;
        for (U32 c = 0; c < 2; ++c)
        {
            const U32 card = c == 0 ? mine.card0 : mine.card1;
            for (U32 entry = cards[opponent * 53 + card]; entry < cards[opponent * 53 + card + 1]; ++entry)
            {
                const U32 other = cards[entry];
                const float r = reach[other];
                if (needMass)
                    blocked += r;
                if (n.kind == NodeKind::Showdown)
                {
                    const U32 otherRank = ranks[U64(n.rankRow) * total + oppBase + other];
                    if (otherRank < rank)
                        blockedWins += r;
                    if (otherRank > rank)
                        blockedLosses += r;
                }
            }
        }
        mass -= blocked;
        if (needMass && mine.matching >= 0)
            mass += reach[mine.matching];
        wins -= blockedWins;
        losses -= blockedLosses;
        output[h] = n.kind == NodeKind::Fold ? (mass / mine.divisor) * (p == 0 ? n.utility[0] : -n.utility[0])
                                             : (tie * mass + (win - tie) * wins + (loss - tie) * losses) / mine.divisor;
    }
}

GPU_ENTRY(Backup)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U32 index = GPU_INDEX, item = index / state.stride, h = index % state.stride;
    const U32 count = state.player ? state.hands[1] : state.hands[0];
    if (item >= pass.count || h >= count)
        return;
    const Node n = nodes[work[pass.offset + item]];
    const bool acting = n.kind == NodeKind::Decision && n.actor == state.player;
    // Reuse the entry-policy denominator for value backup and strategy sums.
    float positive = 0.0f;
    if (acting && !state.evaluation)
        for (U32 a = 0; a < n.count; ++a)
        {
            const float r = regrets[n.strategy + U64(a) * count + h];
            positive += r > 0.0f ? r : 0.0f;
        }
    float value = 0.0f;
    for (U32 a = 0; a < n.count; ++a)
    {
        const float child = values[U64(childSlots[n.edge + a]) * state.stride + h];
        if (acting && state.evaluation)
            value = a == 0 || child > value ? child : value;
        else if (acting)
        {
            const float r = regrets[n.strategy + U64(a) * count + h];
            const float probability = positive > 0.0f ? (r > 0.0f ? r : 0.0f) / positive : 1.0f / float(n.count);
            value += child * probability;
        }
        else
            value += child;
    }
    values[U64(n.slot) * state.stride + h] = value;
    if (acting && !state.evaluation)
    {
        const U32 total = state.hands[0] + state.hands[1], base = state.player ? state.hands[0] : 0;
        const float own = scratch[U64(n.slot) * total + base + h];
        for (U32 a = 0; a < n.count; ++a)
        {
            const U64 entry = n.strategy + U64(a) * count + h;
            const float old = regrets[entry];
            const float probability = positive > 0.0f ? (old > 0.0f ? old : 0.0f) / positive : 1.0f / float(n.count);
            const float child = values[U64(childSlots[n.edge + a]) * state.stride + h];
            const float r = old + (child - value);
            regrets[entry] = r * (r > 0.0f ? state.positiveDiscount : 0.5f);
            sums[entry] = state.averageDiscount * (sums[entry] + own * probability);
        }
    }
}

GPU_ENTRY(Outcomes)(GPU_ARGUMENTS)
{
    GPU_STATE
    const U64 index = GPU_INDEX, pairs = U64(state.hands[0]) * state.hands[1];
    if (index >= pass.count)
        return;
    const U32 row = U32(index / pairs), h0 = U32(index % pairs / state.hands[1]), h1 = U32(index % state.hands[1]);
    const U64 mask0 = hands[h0].mask, mask1 = hands[state.hands[0] + h1].mask;
    const U64 board = nodes[0].board;
    U32 wins = 0, losses = 0;
    if (!(mask0 & mask1))
    {
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
                    const U64 offset = U64(rankRow) * (state.hands[0] + state.hands[1]);
                    const U32 a = ranks[offset + h0], b = ranks[offset + state.hands[0] + h1];
                    wins += a > b;
                    losses += a < b;
                }
            }
        }
    }
    outcomes[index] = wins | (losses << 16);
}
