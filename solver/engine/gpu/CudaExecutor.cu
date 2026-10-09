#include "engine/gpu/GpuPlan.h"
#include "engine/gpu/GpuQuantize.h"
#include <cuda_runtime.h>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include "engine/gpu/GpuKernels.inc"

namespace solver::engine::gpu
{
namespace
{
void Check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}
// Threads of a Runout block for the given lanes, one per two hands: whole warps, at least the two
// FoldMasses splits, and all lanes in one round.
static_assert((HandBoardData::kMaxHands + 1) / 2 <= kRunoutGroup);
U32 RunoutGroup(U32 lanes)
{
    return std::max<U32>((lanes + kWarpSize - 1) / kWarpSize, 2) * kWarpSize;
}
using KernelFunction = void (*)(GPU_ARGUMENTS);
// Terminal's instance and threads per block for a launch with the given shared memory: blocks of
// 2 * kTerminalGroup threads where shared memory, which grows with the opponent's hands, already
// limits resident blocks to what the doubled block allows, so wide ranges' hand loops take half the
// rounds at the same resident blocks.
std::pair<KernelFunction, U32> TerminalInstance(bool averaging, std::size_t shared)
{
    const KernelFunction narrow = averaging ? Terminal<true, kTerminalGroup> : Terminal<false, kTerminalGroup>;
    const KernelFunction wide = averaging ? Terminal<true, 2 * kTerminalGroup> : Terminal<false, 2 * kTerminalGroup>;
    int narrowBlocks = 0, wideBlocks = 0;
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&narrowBlocks, narrow, kTerminalGroup, shared));
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&wideBlocks, wide, 2 * kTerminalGroup, shared));
    return wideBlocks >= narrowBlocks ? std::pair{wide, 2u * kTerminalGroup} : std::pair{narrow, U32(kTerminalGroup)};
}
class CudaExecutor final : public Executor
{
public:
    explicit CudaExecutor(const Plan& plan)
        : shape_(plan.state), entries_(plan.entries), passes_(plan.passes), predecessors_(plan.predecessors)
    {
        try
        {
            Check(cudaSetDevice(0));
            Check(cudaDeviceGetStreamPriorityRange(&lowPriority_, &highPriority_));
            // The kernels wait for their predecessors only when built for compute capability 9 or
            // later; a device without such code JIT-compiles older PTX.
            cudaFuncAttributes attributes{};
            Check(cudaFuncGetAttributes(&attributes, Reach));
            programmatic_ = attributes.ptxVersion >= 90;
            for (auto& stream : streams_)
                Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            Check(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
            Check(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming));
            events_.resize(plan.passes.size());
            for (auto& event : events_)
                Check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            for (auto& event : done_)
                Check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            Check(cudaHostAlloc(reinterpret_cast<void**>(&staging_), done_.size() * sizeof(State), cudaHostAllocDefault));
            const auto sources = plan.Buffers();
            for (std::size_t i = 0; i < buffers_.size(); ++i)
                Upload(i, sources[i]);
            PersistScratch(sources[ScratchBuffer].bytes);
            for (const auto index : {RegretsBuffer, SumsBuffer, FlagsBuffer, MasksBuffer, StampsBuffer})
                Check(cudaMemsetAsync(buffers_[index], 0, sources[index].bytes, stream_));
            for (const auto& pass : plan.initialization)
                Dispatch(pass, stream_);
            Check(cudaStreamSynchronize(stream_));
            for (U32 player = 0; player < 2; ++player)
                Capture(player, false);
        }
        catch (...)
        {
            Release();
            throw;
        }
    }
    ~CudaExecutor() override { Release(); }
    const char* Name() const override { return "CUDA"; }
    void Update(const State& state) override
    {
        // Updates queue behind each other in the origin stream, at most two in flight, so
        // the host submits the next graph while the device runs the current one instead of
        // idling the device for the submission. The pinned staging slot is reused once the
        // update that copied from it has finished; an unrecorded event has nothing to wait for.
        Check(cudaEventSynchronize(done_[slot_]));
        staging_[slot_] = state;
        Check(cudaMemcpyAsync(buffers_[StateBuffer], &staging_[slot_], sizeof(State), cudaMemcpyHostToDevice, stream_));
        // Strategy values run the Averaging kernels (see Terminal) in graphs of their own, captured
        // at first use.
        const bool averaging = state.evaluation == Evaluation::StrategyValue;
        if (!executables_[averaging][state.player])
            Capture(state.player, averaging);
        Check(cudaGraphLaunch(executables_[averaging][state.player], stream_));
        Check(cudaEventRecord(done_[slot_], stream_));
        slot_ = (slot_ + 1) % done_.size();
    }
    void Synchronize() override { Check(cudaStreamSynchronize(stream_)); }
    std::vector<float> RootValues(const State& state) override
    {
        Update(state);
        std::vector<float> values(state.hands[state.player]);
        Copy(values.data(), buffers_[ScratchBuffer], values.size() * sizeof(float), cudaMemcpyDeviceToHost);
        return values;
    }
    void StreamSums(const std::vector<std::size_t>& ends, const std::function<void(const std::uint16_t*)>& consume) override
    {
        Synchronize();
        DestroyGraph();
        for (std::size_t i = 0; i < buffers_.size(); ++i)
            if (i != SumsBuffer)
                Free(i);
        // Ranges alternate between two pinned halves (one for a single range), each copy marked by
        // an update event, idle now.
        std::size_t half = 0;
        for (std::size_t k = 0, begin = 0; k < ends.size(); begin = ends[k++])
            half = std::max(half, ends[k] - begin);
        const std::size_t halves = ends.size() > 1 ? 2 : 1;
        Check(cudaHostAlloc(reinterpret_cast<void**>(&download_), halves * half * sizeof(std::uint16_t), cudaHostAllocDefault));
        const auto copy = [&](std::size_t k)
        {
            const std::size_t begin = k ? ends[k - 1] : 0;
            Check(cudaMemcpyAsync(
                download_ + k % 2 * half,
                static_cast<const std::uint16_t*>(buffers_[SumsBuffer]) + begin,
                (ends[k] - begin) * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost,
                stream_
            ));
            Check(cudaEventRecord(done_[k % 2], stream_));
        };
        copy(0);
        for (std::size_t k = 0; k < ends.size(); ++k)
        {
            if (k + 1 < ends.size())
                copy(k + 1);
            Check(cudaEventSynchronize(done_[k % 2]));
            consume(download_ + k % 2 * half);
        }
        Check(cudaFreeHost(download_));
        download_ = nullptr;
        Free(SumsBuffer);
    }
    QuantizedState DownloadTraining() override
    {
        QuantizedState state{std::vector<std::int16_t>(entries_), std::vector<std::uint16_t>(entries_)};
        Copy(state.regrets.data(), buffers_[RegretsBuffer], entries_ * sizeof(std::int16_t), cudaMemcpyDeviceToHost);
        Copy(state.strategySums.data(), buffers_[SumsBuffer], entries_ * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        std::vector<std::uint32_t> halves(2 * std::size_t(shape_.stampCount));
        Copy(halves.data(), buffers_[StampsBuffer], halves.size() * sizeof(U32), cudaMemcpyDeviceToHost);
        state.stamps = NewestStamps(halves);
        return state;
    }
    void UploadTraining(const QuantizedState& state) override
    {
        Copy(buffers_[RegretsBuffer], state.regrets.data(), entries_ * sizeof(std::int16_t), cudaMemcpyHostToDevice);
        Copy(buffers_[SumsBuffer], state.strategySums.data(), entries_ * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
        for (std::size_t half = 0; half < 2; ++half)
            Copy(
                static_cast<U32*>(buffers_[StampsBuffer]) + half * shape_.stampCount,
                state.stamps.data(),
                shape_.stampCount * sizeof(U32),
                cudaMemcpyHostToDevice
            );
    }

private:
    std::array<void*, kBufferCount> buffers_{};
    State shape_;
    std::size_t entries_;
    std::vector<Pass> passes_;
    std::vector<std::vector<U32>> predecessors_;
    // One stream per Pass::lane; the first is the capture origin and update stream. Each
    // pass records an event so passes in other streams can wait for their predecessors.
    std::array<cudaStream_t, kLaneCount> streams_{};
    cudaStream_t& stream_ = streams_[0];
    cudaEvent_t fork_ = nullptr; // the origin's state at the start of a capture
    cudaEvent_t join_ = nullptr; // each other stream's end, joined into the origin
    std::vector<cudaEvent_t> events_;
    // Pinned State copies of the updates in flight and the events ending them.
    State* staging_ = nullptr;
    std::uint16_t* download_ = nullptr; // StreamSums's pinned ranges
    std::array<cudaEvent_t, 2> done_{};
    std::size_t slot_ = 0;
    // Per mode, training and best response or else Averaging, and per player.
    std::array<std::array<cudaGraph_t, 2>, 2> graphs_{};
    std::array<std::array<cudaGraphExec_t, 2>, 2> executables_{};
    int lowPriority_ = 0, highPriority_ = 0;
    // Whether captured passes launch programmatically (see Dispatch): kernels built for compute capability 9 and later.
    bool programmatic_ = false;
    // The launches' L2 access policy over the scratch buffer (see PersistScratch); an empty window
    // when the device has no persisting L2.
    cudaAccessPolicyWindow scratchWindow_{};
    void Upload(std::size_t index, const BufferData& source)
    {
        Check(cudaMalloc(&buffers_[index], source.AllocationBytes()));
        if (source.data)
            Copy(buffers_[index], source.data, source.bytes, cudaMemcpyHostToDevice);
    }
    // Batches rewrite their lanes' scratch rows every few passes, yet with the training state
    // streaming through L2 (see LoadStreamed in GpuKernels.inc) many dirty rows still leave it
    // between passes: a persisting share of the buffer's lines, spread over all of it, stays
    // resident instead of being written back and refetched. A set-aside of 5/16 of the L2 (20 of
    // 64 MB) measured fastest; larger ones crowd out the rows and records that remain normal.
    void PersistScratch(std::size_t bytes)
    {
        int l2 = 0, persisting = 0, windowLimit = 0;
        Check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
        Check(cudaDeviceGetAttribute(&persisting, cudaDevAttrMaxPersistingL2CacheSize, 0));
        Check(cudaDeviceGetAttribute(&windowLimit, cudaDevAttrMaxAccessPolicyWindowSize, 0));
        const std::size_t setAside = std::min<std::size_t>(persisting, std::size_t(l2) / 16 * 5);
        if (!setAside || !windowLimit || !bytes)
            return;
        Check(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, setAside));
        scratchWindow_.base_ptr = buffers_[ScratchBuffer];
        scratchWindow_.num_bytes = std::min<std::size_t>(bytes, windowLimit);
        scratchWindow_.hitRatio = std::min(1.0f, float(setAside) / float(scratchWindow_.num_bytes));
        scratchWindow_.hitProp = cudaAccessPropertyPersisting;
        scratchWindow_.missProp = cudaAccessPropertyNormal;
    }
    void Copy(void* target, const void* source, std::size_t bytes, cudaMemcpyKind kind)
    {
        // The non-blocking stream does not wait for legacy-stream copies, and pageable
        // uploads may return before their DMA completes.
        if (!bytes)
            return;
        Check(cudaMemcpyAsync(target, source, bytes, kind, stream_));
        Check(cudaStreamSynchronize(stream_));
    }
    // Captures one player's update, or with averaging its strategy-value evaluation: every stream
    // forks from the origin at its first pass, waits for the events of the pass's predecessors, and
    // joins the origin at the end. Each launch moves the node records of the next launch in its
    // stream into L2, which every record would otherwise wait for in DRAM.
    void Capture(U32 player, bool averaging)
    {
        std::vector<Pass> launches(passes_.size());
        std::array<const Pass*, kLaneCount> following{};
        for (std::size_t i = passes_.size(); i-- > 0;)
        {
            Pass& launch = launches[i] = LaunchPass(passes_[i], shape_, player);
            if (launch.count == 0)
                continue;
            if (const Pass* next = following[launch.lane])
            {
                launch.next = next->offset;
                launch.nextCount = next->count;
            }
            following[launch.lane] = &launch;
        }
        Check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        Check(cudaEventRecord(fork_, stream_));
        std::array<bool, kLaneCount> used{true};
        for (std::size_t i = 0; i < passes_.size(); ++i)
        {
            const Pass& pass = launches[i];
            cudaStream_t stream = streams_[pass.lane];
            if (!used[pass.lane])
            {
                Check(cudaStreamWaitEvent(stream, fork_, 0));
                used[pass.lane] = true;
            }
            for (const U32 predecessor : predecessors_[i])
                Check(cudaStreamWaitEvent(stream, events_[predecessor], 0));
            Dispatch(pass, stream, averaging, programmatic_);
            Check(cudaEventRecord(events_[i], stream));
        }
        for (std::size_t lane = 1; lane < streams_.size(); ++lane)
            if (used[lane])
            {
                Check(cudaEventRecord(join_, streams_[lane]));
                Check(cudaStreamWaitEvent(stream_, join_, 0));
            }
        auto& graph = graphs_[averaging][player];
        Check(cudaStreamEndCapture(stream_, &graph));
        Check(cudaGraphInstantiateWithFlags(&executables_[averaging][player], graph, cudaGraphInstantiateFlagUseNodePriority));
    }
    void Dispatch(const Pass& pass, cudaStream_t stream, bool averaging = false, bool programmatic = false)
    {
        if (pass.count == 0)
            return;
        // Reach and Backup tile each item's pass.lanes, two hands per lane, Terminal runs one group per item (see
        // TerminalInstance) and Runout one block (see RunoutGroup); Outcomes is linear over pass.lanes.
        KernelFunction kernel = Outcomes;
        U32 group = 256;
        std::size_t shared = 0;
        dim3 blocks(pass.count);
        switch (pass.operation)
        {
        case Kernel::Reach:
        case Kernel::Backup:
            kernel = pass.operation == Kernel::Reach ? Reach : averaging ? Backup<true> : Backup<false>;
            group = (std::min<U32>(pass.lanes, kTileGroup) + kWarpSize - 1) / kWarpSize * kWarpSize;
            blocks = dim3(pass.count, (pass.lanes + group - 1) / group);
            break;
        case Kernel::Terminal:
        {
            shared = TerminalLayout(pass.lanes).end * sizeof(float);
            const auto instance = TerminalInstance(averaging, shared);
            kernel = instance.first;
            group = instance.second;
            break;
        }
        case Kernel::Runout:
            kernel = Runout;
            group = RunoutGroup(pass.lanes);
            shared = RunoutShared(shape_.stride) * sizeof(float);
            break;
        case Kernel::Outcomes:
            blocks = dim3((pass.count * pass.lanes + group - 1) / group);
            break;
        }
        const bool leaves = pass.operation == Kernel::Terminal || pass.operation == Kernel::Runout;
        // Leaf batches' Terminal and Runout blocks run at the lowest priority: freed SM slots go first to pending Reach and
        // Backup blocks, which gate their streams' next passes, and long Terminal launches fill the rest instead of holding
        // every SM while the other streams wait (7% faster). The spine's terminal passes gate the spine's Backup instead
        // and keep the highest priority.
        cudaLaunchAttribute attributes[3]{};
        unsigned int count = 0;
        attributes[count].id = cudaLaunchAttributePriority;
        attributes[count++].val.priority = leaves && pass.lane < kSpineLane ? lowPriority_ : highPriority_;
        if (scratchWindow_.num_bytes)
        {
            attributes[count].id = cudaLaunchAttributeAccessPolicyWindow;
            attributes[count++].val.accessPolicyWindow = scratchWindow_;
        }
        // A programmatic launch's blocks may start before its predecessors finish; each kernel waits
        // for them (WaitForPredecessors in GpuKernels.inc) after loading its State and node record.
        if (programmatic)
        {
            attributes[count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attributes[count++].val.programmaticStreamSerializationAllowed = 1;
        }
        cudaLaunchConfig_t config{};
        config.gridDim = blocks;
        config.blockDim = dim3(group);
        config.dynamicSmemBytes = shared;
        config.stream = stream;
        config.attrs = attributes;
        config.numAttrs = count;
        Check(cudaLaunchKernelEx(
            &config,
            kernel,
            static_cast<const Node*>(buffers_[NodesBuffer]),
            static_cast<const Hand*>(buffers_[HandsBuffer]),
            static_cast<const float*>(buffers_[WeightsBuffer]),
            static_cast<const unsigned short*>(buffers_[RanksBuffer]),
            static_cast<const int*>(buffers_[RunoutsBuffer]),
            static_cast<const U32*>(buffers_[OrderBuffer]),
            static_cast<const U32*>(buffers_[CardsBuffer]),
            static_cast<U32*>(buffers_[OutcomesBuffer]),
            static_cast<short*>(buffers_[RegretsBuffer]),
            static_cast<unsigned short*>(buffers_[SumsBuffer]),
            static_cast<float*>(buffers_[ScratchBuffer]),
            static_cast<U32*>(buffers_[FlagsBuffer]),
            static_cast<U32*>(buffers_[MasksBuffer]),
            static_cast<U32*>(buffers_[StampsBuffer]),
            static_cast<const State*>(buffers_[StateBuffer]),
            pass
        ));
    }
    void Free(std::size_t i)
    {
        if (buffers_[i])
            cudaFree(buffers_[i]);
        buffers_[i] = nullptr;
    }
    void DestroyGraph()
    {
        for (std::size_t mode = 0; mode < 2; ++mode)
            for (U32 player = 0; player < 2; ++player)
            {
                if (executables_[mode][player])
                    cudaGraphExecDestroy(executables_[mode][player]);
                if (graphs_[mode][player])
                    cudaGraphDestroy(graphs_[mode][player]);
                executables_[mode][player] = nullptr;
                graphs_[mode][player] = nullptr;
            }
    }
    void Release()
    {
        if (stream_)
            cudaStreamSynchronize(stream_);
        DestroyGraph();
        for (std::size_t i = 0; i < buffers_.size(); ++i)
            Free(i);
        if (scratchWindow_.num_bytes)
        {
            // Later work in the process gets the whole L2 back.
            cudaCtxResetPersistingL2Cache();
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
            scratchWindow_ = {};
        }
        for (auto& stream : streams_)
            if (stream)
                cudaStreamDestroy(stream);
        if (fork_)
            cudaEventDestroy(fork_);
        if (join_)
            cudaEventDestroy(join_);
        for (auto& event : events_)
            if (event)
                cudaEventDestroy(event);
        for (auto& event : done_)
            if (event)
                cudaEventDestroy(event);
        if (staging_)
            cudaFreeHost(staging_);
        if (download_)
            cudaFreeHost(download_);
        streams_ = {};
        fork_ = join_ = nullptr;
        events_.clear();
        done_ = {};
        staging_ = nullptr;
        download_ = nullptr;
    }
};
} // namespace
bool DeviceAvailable()
{
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess)
        cudaGetLastError();
    if (status != cudaSuccess || count == 0)
        return false;
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, 0) == cudaSuccess && properties.major * 10 + properties.minor >= 86;
}
std::uint64_t DeviceMemoryBudget()
{
    Check(cudaSetDevice(0));
    std::size_t available, total;
    Check(cudaMemGetInfo(&available, &total));
    return available;
}
std::unique_ptr<Executor> MakeExecutor(const Plan& plan)
{
    return std::make_unique<CudaExecutor>(plan);
}
} // namespace solver::engine::gpu
