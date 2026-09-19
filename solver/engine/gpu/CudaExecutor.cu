#include "engine/gpu/GpuPlan.h"
#include <cuda_runtime.h>
#include <algorithm>
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
            Upload(0, plan.nodes);
            Upload(1, plan.edges);
            Upload(2, plan.hands);
            Upload(3, plan.ranks);
            Upload(4, plan.runouts);
            Upload(5, plan.order);
            Upload(6, plan.cards);
            Upload(7, plan.work);
            Allocate(8, plan.outcomeEntries * sizeof(U32));
            Allocate(9, entries_ * sizeof(float));
            Allocate(10, entries_ * sizeof(float));
            Allocate(11, plan.slots * (shape_.hands[0] + shape_.hands[1]) * sizeof(float));
            Allocate(12, plan.slots * 3 * shape_.stride * sizeof(float));
            Allocate(13, sizeof(State));
            Check(cudaMemsetAsync(buffers_[9], 0, entries_ * sizeof(float), stream_));
            Check(cudaMemsetAsync(buffers_[10], 0, entries_ * sizeof(float), stream_));
            Check(cudaMemcpyAsync(buffers_[13], &shape_, sizeof(State), cudaMemcpyHostToDevice, stream_));
            if (shape_.outcomeRows)
            {
                Launch({Kernel::Outcomes, 0, static_cast<U32>(plan.outcomeEntries)});
                if (shape_.outcomeRows > 1)
                    Launch({Kernel::Outcomes, 1, shape_.hands[0] * shape_.hands[1]});
            }
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
        Check(cudaMemcpyAsync(buffers_[13], &state, sizeof(State), cudaMemcpyHostToDevice, stream_));
        Check(cudaGraphLaunch(executable_, stream_));
        // One synchronization per player update, never per node or batch.
        Check(cudaStreamSynchronize(stream_));
    }
    std::vector<float> RootValues(const State& state) override
    {
        Update(state);
        std::vector<float> values(state.hands[state.player]);
        Check(cudaMemcpy(values.data(), buffers_[12], values.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return values;
    }
    std::vector<float> DownloadSums(bool releaseTraining) override
    {
        if (releaseTraining)
        {
            DestroyGraph();
            for (std::size_t i = 0; i < buffers_.size(); ++i)
                if (i != 10)
                    Free(i);
        }
        std::vector<float> result(entries_);
        if (!result.empty())
            Check(cudaMemcpy(result.data(), buffers_[10], result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (releaseTraining)
            Free(10);
        return result;
    }

private:
    std::array<void*, 14> buffers_{};
    State shape_;
    std::size_t entries_;
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    void Allocate(std::size_t index, std::size_t bytes) { Check(cudaMalloc(&buffers_[index], std::max<std::size_t>(bytes, 4))); }
    template<typename T>
    void Upload(std::size_t index, const std::vector<T>& source)
    {
        Allocate(index, source.size() * sizeof(T));
        if (!source.empty())
            Check(cudaMemcpy(buffers_[index], source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void Launch(Pass pass)
    {
        const auto threads = pass.count * (pass.kernel == Kernel::Prefix || pass.kernel == Kernel::Outcomes ? 1 : shape_.stride);
        const auto blocks = (threads + 255) / 256;
#define LAUNCH(name)                                     \
    name<<<blocks, 256, 0, stream_>>>(                   \
        static_cast<const Node*>(buffers_[0]),           \
        static_cast<const U32*>(buffers_[1]),            \
        static_cast<const Hand*>(buffers_[2]),           \
        static_cast<const unsigned short*>(buffers_[3]), \
        static_cast<const int*>(buffers_[4]),            \
        static_cast<const U32*>(buffers_[5]),            \
        static_cast<const U32*>(buffers_[6]),            \
        static_cast<const U32*>(buffers_[7]),            \
        static_cast<U32*>(buffers_[8]),                  \
        static_cast<float*>(buffers_[9]),                \
        static_cast<float*>(buffers_[10]),               \
        static_cast<float*>(buffers_[11]),               \
        static_cast<float*>(buffers_[12]),               \
        static_cast<const State*>(buffers_[13]),         \
        pass                                             \
    )
        switch (pass.kernel)
        {
        case Kernel::Reach:
            LAUNCH(Reach);
            break;
        case Kernel::Prefix:
            LAUNCH(Prefix);
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
