#include "engine/gpu/GpuPlan.h"
#include "engine/gpu/GpuQuantize.h"
#include <cuda_runtime.h>
#include <array>
#include <stdexcept>
#include <string>
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
class CudaExecutor final : public Executor
{
public:
    explicit CudaExecutor(const Plan& plan) : shape_(plan.state), entries_(plan.entries)
    {
        try
        {
            Check(cudaSetDevice(0));
            Check(cudaDeviceGetStreamPriorityRange(&lowPriority_, &highPriority_));
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
            for (const auto index : {RegretsBuffer, SumsBuffer, FlagsBuffer, StampsBuffer})
                Check(cudaMemsetAsync(buffers_[index], 0, sources[index].bytes, stream_));
            for (const auto& pass : plan.initialization)
                Dispatch(pass, stream_);
            Check(cudaStreamSynchronize(stream_));
            for (U32 player = 0; player < 2; ++player)
                Capture(plan, player);
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
        Check(cudaGraphLaunch(executables_[state.player], stream_));
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
    std::vector<std::uint16_t> DownloadSums(bool releaseTraining) override
    {
        if (releaseTraining)
        {
            Synchronize();
            DestroyGraph();
            for (std::size_t i = 0; i < buffers_.size(); ++i)
                if (i != SumsBuffer)
                    Free(i);
        }
        std::vector<std::uint16_t> result(entries_);
        Copy(result.data(), buffers_[SumsBuffer], result.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        if (releaseTraining)
            Free(SumsBuffer);
        return result;
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
    // One stream per Pass::lane; the first is the capture origin and update stream. Each
    // pass records an event so passes in other streams can wait for their predecessors.
    std::array<cudaStream_t, kLaneCount> streams_{};
    cudaStream_t& stream_ = streams_[0];
    cudaEvent_t fork_ = nullptr; // the origin's state at the start of a capture
    cudaEvent_t join_ = nullptr; // each other stream's end, joined into the origin
    std::vector<cudaEvent_t> events_;
    // Pinned State copies of the updates in flight and the events ending them.
    State* staging_ = nullptr;
    std::array<cudaEvent_t, 2> done_{};
    std::size_t slot_ = 0;
    std::array<cudaGraph_t, 2> graphs_{};
    std::array<cudaGraphExec_t, 2> executables_{};
    int lowPriority_ = 0, highPriority_ = 0;
    void Upload(std::size_t index, const BufferData& source)
    {
        Check(cudaMalloc(&buffers_[index], source.AllocationBytes()));
        if (source.data)
            Copy(buffers_[index], source.data, source.bytes, cudaMemcpyHostToDevice);
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
    // Captures one player's update: every stream forks from the origin at its first pass,
    // waits for the events of the pass's predecessors, and joins the origin at the end.
    void Capture(const Plan& plan, U32 player)
    {
        Check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        Check(cudaEventRecord(fork_, stream_));
        std::array<bool, kLaneCount> used{true};
        for (std::size_t i = 0; i < plan.passes.size(); ++i)
        {
            const Pass pass = LaunchPass(plan.passes[i], shape_, player);
            cudaStream_t stream = streams_[pass.lane];
            if (!used[pass.lane])
            {
                Check(cudaStreamWaitEvent(stream, fork_, 0));
                used[pass.lane] = true;
            }
            for (const U32 predecessor : plan.predecessors[i])
                Check(cudaStreamWaitEvent(stream, events_[predecessor], 0));
            Dispatch(pass, stream);
            Check(cudaEventRecord(events_[i], stream));
        }
        for (std::size_t lane = 1; lane < streams_.size(); ++lane)
            if (used[lane])
            {
                Check(cudaEventRecord(join_, streams_[lane]));
                Check(cudaStreamWaitEvent(stream_, join_, 0));
            }
        Check(cudaStreamEndCapture(stream_, &graphs_[player]));
        Check(cudaGraphInstantiateWithFlags(&executables_[player], graphs_[player], cudaGraphInstantiateFlagUseNodePriority));
    }
    void Dispatch(const Pass& pass, cudaStream_t stream)
    {
        if (pass.count == 0)
            return;
        // Reach and Backup tile each item's pass.lanes, two hands per lane, and Terminal runs one group per item; other
        // passes are linear over pass.lanes.
        const bool handTiles = pass.operation == Kernel::Backup || pass.operation == Kernel::Reach;
        const U32 group = pass.operation == Kernel::Terminal ? kTerminalGroup
                          : handTiles ? (std::min<U32>(pass.lanes, kTileGroup) + kWarpSize - 1) / kWarpSize * kWarpSize
                                      : 256u;
        const auto threads = pass.count * pass.lanes;
        const dim3 blocks = handTiles ? dim3(pass.count, (pass.lanes + group - 1) / group)
                                      : dim3(pass.operation == Kernel::Terminal ? pass.count : (threads + group - 1) / group);
        const auto shared = pass.operation == Kernel::Terminal ? TerminalLayout(pass.lanes).end * sizeof(float) : 0;
        // Terminal blocks run at the lowest priority: freed SM slots go first to pending Reach and
        // Backup blocks, which gate their streams' next passes, and long Terminal launches fill
        // the rest instead of holding every SM while the other streams wait (7% faster).
        cudaLaunchAttribute priority{};
        priority.id = cudaLaunchAttributePriority;
        priority.val.priority = pass.operation == Kernel::Terminal ? lowPriority_ : highPriority_;
        cudaLaunchConfig_t config{};
        config.gridDim = blocks;
        config.blockDim = dim3(group);
        config.dynamicSmemBytes = shared;
        config.stream = stream;
        config.attrs = &priority;
        config.numAttrs = 1;
        const auto kernel = pass.operation == Kernel::Reach      ? Reach
                            : pass.operation == Kernel::Terminal ? Terminal
                            : pass.operation == Kernel::Backup   ? Backup
                                                                 : Outcomes;
        Check(cudaLaunchKernelEx(
            &config,
            kernel,
            static_cast<const Node*>(buffers_[NodesBuffer]),
            static_cast<const Hand*>(buffers_[HandsBuffer]),
            static_cast<const unsigned short*>(buffers_[RanksBuffer]),
            static_cast<const int*>(buffers_[RunoutsBuffer]),
            static_cast<const U32*>(buffers_[OrderBuffer]),
            static_cast<const U32*>(buffers_[CardsBuffer]),
            static_cast<U32*>(buffers_[OutcomesBuffer]),
            static_cast<short*>(buffers_[RegretsBuffer]),
            static_cast<unsigned short*>(buffers_[SumsBuffer]),
            static_cast<float*>(buffers_[ScratchBuffer]),
            static_cast<U32*>(buffers_[FlagsBuffer]),
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
        for (U32 player = 0; player < 2; ++player)
        {
            if (executables_[player])
                cudaGraphExecDestroy(executables_[player]);
            if (graphs_[player])
                cudaGraphDestroy(graphs_[player]);
            executables_[player] = nullptr;
            graphs_[player] = nullptr;
        }
    }
    void Release()
    {
        if (stream_)
            cudaStreamSynchronize(stream_);
        DestroyGraph();
        for (std::size_t i = 0; i < buffers_.size(); ++i)
            Free(i);
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
        streams_ = {};
        fork_ = join_ = nullptr;
        events_.clear();
        done_ = {};
        staging_ = nullptr;
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
