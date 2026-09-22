#include "engine/gpu/GpuPlan.h"
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
            std::size_t available, total;
            Check(cudaMemGetInfo(&available, &total));
            if (plan.DeviceBytes() > available)
                throw std::runtime_error("GPU training state and batch scratch exceed available CUDA memory");
            Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            const auto sources = plan.Buffers();
            for (std::size_t i = 0; i < buffers_.size(); ++i)
                Upload(i, sources[i]);
            Check(cudaMemsetAsync(buffers_[RegretsBuffer], 0, sources[RegretsBuffer].bytes, stream_));
            Check(cudaMemsetAsync(buffers_[SumsBuffer], 0, sources[SumsBuffer].bytes, stream_));
            for (const auto& pass : plan.initialization)
                Launch(pass);
            Check(cudaStreamSynchronize(stream_));
            Check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            for (const auto& pass : plan.passes)
                Launch(pass);
            Check(cudaStreamEndCapture(stream_, &graph_));
            Check(cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0));
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
        Check(cudaMemcpyAsync(buffers_[StateBuffer], &state, sizeof(State), cudaMemcpyHostToDevice, stream_));
        Check(cudaGraphLaunch(executable_, stream_));
        // One synchronization per player update, never per node or batch.
        Check(cudaStreamSynchronize(stream_));
    }
    std::vector<float> RootValues(const State& state) override
    {
        Update(state);
        std::vector<float> values(state.hands[state.player]);
        Check(cudaMemcpy(values.data(), buffers_[ValuesBuffer], values.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return values;
    }
    std::vector<float> DownloadSums(bool releaseTraining) override
    {
        if (releaseTraining)
        {
            DestroyGraph();
            for (std::size_t i = 0; i < buffers_.size(); ++i)
                if (i != SumsBuffer)
                    Free(i);
        }
        std::vector<float> result(entries_);
        if (!result.empty())
            Check(cudaMemcpy(result.data(), buffers_[SumsBuffer], result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (releaseTraining)
            Free(SumsBuffer);
        return result;
    }

private:
    std::array<void*, kBufferCount> buffers_{};
    State shape_;
    std::size_t entries_;
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    void Upload(std::size_t index, const BufferData& source)
    {
        Check(cudaMalloc(&buffers_[index], source.AllocationBytes()));
        if (source.data && source.bytes)
            Check(cudaMemcpy(buffers_[index], source.data, source.bytes, cudaMemcpyHostToDevice));
    }
    void Launch(Pass pass)
    {
        const auto threads = pass.count * (pass.operation == Kernel::Outcomes ? 1 : shape_.stride);
        const auto group = pass.operation == Kernel::Terminal ? 128 : 256;
        const auto blocks = pass.operation == Kernel::Terminal ? pass.count : (threads + group - 1) / group;
        const auto shared = pass.operation == Kernel::Terminal ? TerminalSharedBytes(shape_) : 0;
#define LAUNCH(name)                                               \
    name<<<blocks, group, shared, stream_>>>(                      \
        static_cast<const Node*>(buffers_[NodesBuffer]),           \
        static_cast<const U32*>(buffers_[ChildSlotsBuffer]),       \
        static_cast<const Hand*>(buffers_[HandsBuffer]),           \
        static_cast<const unsigned short*>(buffers_[RanksBuffer]), \
        static_cast<const int*>(buffers_[RunoutsBuffer]),          \
        static_cast<const U32*>(buffers_[OrderBuffer]),            \
        static_cast<const U32*>(buffers_[CardsBuffer]),            \
        static_cast<const U32*>(buffers_[WorkBuffer]),             \
        static_cast<U32*>(buffers_[OutcomesBuffer]),               \
        static_cast<float*>(buffers_[RegretsBuffer]),              \
        static_cast<float*>(buffers_[SumsBuffer]),                 \
        static_cast<float*>(buffers_[ScratchBuffer]),              \
        static_cast<float*>(buffers_[ValuesBuffer]),               \
        static_cast<const State*>(buffers_[StateBuffer]),          \
        pass                                                       \
    )
        switch (pass.operation)
        {
        case Kernel::Reach:
            LAUNCH(Reach);
            break;
        case Kernel::Terminal:
            LAUNCH(Terminal);
            break;
        case Kernel::Backup:
            LAUNCH(Backup);
            break;
        case Kernel::Outcomes:
            LAUNCH(Outcomes);
            break;
        }
#undef LAUNCH
        Check(cudaGetLastError());
    }
    void Free(std::size_t i)
    {
        if (buffers_[i])
            cudaFree(buffers_[i]);
        buffers_[i] = nullptr;
    }
    void DestroyGraph()
    {
        if (executable_)
            cudaGraphExecDestroy(executable_);
        if (graph_)
            cudaGraphDestroy(graph_);
        executable_ = nullptr;
        graph_ = nullptr;
    }
    void Release()
    {
        DestroyGraph();
        for (std::size_t i = 0; i < buffers_.size(); ++i)
            Free(i);
        if (stream_)
            cudaStreamDestroy(stream_);
        stream_ = nullptr;
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
    return cudaGetDeviceProperties(&properties, 0) == cudaSuccess && properties.major * 10 + properties.minor >= 89;
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
