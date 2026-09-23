#include "engine/gpu/GpuPlan.h"
#include "MetalSource.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace solver::engine::gpu
{
namespace
{
void Check(id object, NSError* error, const char* fallback)
{
    if (!object)
        throw std::runtime_error(error ? error.localizedDescription.UTF8String : fallback);
}
class MetalExecutor final : public Executor
{
public:
    explicit MetalExecutor(const Plan& plan) : passes_(plan.passes), shape_(plan.state), entries_(plan.entries)
    {
        @autoreleasepool
        {
            device_ = MTLCreateSystemDefaultDevice();
            Check(device_, nil, "Metal device is unavailable");
            if (plan.DeviceBytes() > device_.recommendedMaxWorkingSetSize)
                throw std::runtime_error("GPU training state and batch scratch exceed the recommended Metal working set");
            queue_ = [device_ newCommandQueue];
            Check(queue_, nil, "Cannot create Metal command queue");
            NSError* error = nil;
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.fastMathEnabled = NO;
            options.languageVersion = MTLLanguageVersion2_3;
            id<MTLLibrary> library = [device_ newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                                           options:options
                                                             error:&error];
            Check(library, error, "Cannot compile DCFR Metal kernels");
            const char* names[] = {"Reach", "Terminal", "Backup", "Outcomes"};
            for (std::size_t i = 0; i < pipelines_.size(); ++i)
            {
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
                Check(function, nil, "Missing DCFR Metal kernel");
                pipelines_[i] = [device_ newComputePipelineStateWithFunction:function error:&error];
                Check(pipelines_[i], error, "Cannot create DCFR Metal pipeline");
            }
            const auto sources = plan.Buffers();
            for (std::size_t i = 0; i < buffers_.size(); ++i)
                Upload(i, sources[i]);
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            [blit fillBuffer:buffers_[RegretsBuffer] range:NSMakeRange(0, buffers_[RegretsBuffer].length) value:0];
            [blit fillBuffer:buffers_[SumsBuffer] range:NSMakeRange(0, buffers_[SumsBuffer].length) value:0];
            [blit endEncoding];
            if (!plan.initialization.empty())
            {
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                Bind(encoder, shape_);
                for (const auto& pass : plan.initialization)
                    Encode(encoder, pass);
                [encoder endEncoding];
            }
            Finish(command);
        }
    }
    const char* Name() const override { return "Metal"; }
    void Update(const State& state) override
    {
        @autoreleasepool
        {
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            Bind(encoder, state);
            for (const auto& pass : passes_)
                Encode(encoder, pass);
            [encoder endEncoding];
            Finish(command);
        }
    }
    std::vector<float> RootValues(const State& state) override
    {
        Update(state);
        return Read(buffers_[ValuesBuffer], state.hands[state.player]);
    }
    std::vector<float> DownloadSums(bool releaseTraining) override
    {
        if (releaseTraining)
            for (std::size_t i = 0; i < buffers_.size(); ++i)
                if (i != SumsBuffer)
                    buffers_[i] = nil;
        auto result = Read(buffers_[SumsBuffer], entries_);
        if (releaseTraining)
            buffers_[SumsBuffer] = nil;
        return result;
    }
    TrainingState DownloadTraining() override { return {Read(buffers_[RegretsBuffer], entries_), Read(buffers_[SumsBuffer], entries_)}; }
    void UploadTraining(const TrainingState& state) override
    {
        Write(buffers_[RegretsBuffer], state.regrets);
        Write(buffers_[SumsBuffer], state.strategySums);
    }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    std::array<id<MTLBuffer>, kDataBufferCount> buffers_{};
    std::array<id<MTLComputePipelineState>, 4> pipelines_{};
    std::vector<Pass> passes_;
    State shape_;
    std::size_t entries_;
    void Upload(std::size_t i, const BufferData& source)
    {
        const auto bytes = source.AllocationBytes();
        if (bytes > device_.maxBufferLength)
            throw std::runtime_error("A DCFR buffer exceeds Metal maxBufferLength");
        const bool upload = source.data && source.bytes;
        const auto options = upload ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
        buffers_[i] = [device_ newBufferWithLength:bytes options:options];
        Check(buffers_[i], nil, "Cannot allocate DCFR Metal buffer");
        if (upload)
            std::memcpy(buffers_[i].contents, source.data, source.bytes);
    }
    void Bind(id<MTLComputeCommandEncoder> encoder, const State& state)
    {
        for (std::size_t i = 0; i < buffers_.size(); ++i)
            [encoder setBuffer:buffers_[i] offset:0 atIndex:i];
        [encoder setBytes:&state length:sizeof(state) atIndex:StateBuffer];
    }
    void Encode(id<MTLComputeCommandEncoder> encoder, Pass pass)
    {
        id<MTLComputePipelineState> pipeline = pipelines_[static_cast<std::size_t>(pass.operation)];
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&pass length:sizeof(pass) atIndex:PassBuffer];
        const auto width = pipeline.threadExecutionWidth;
        const auto target = pass.operation == Kernel::Terminal ? 128 : 256;
        const auto group = std::min<NSUInteger>(target, pipeline.maxTotalThreadsPerThreadgroup) / width * width;
        if (pass.operation == Kernel::Terminal)
        {
            [encoder setThreadgroupMemoryLength:(TerminalSharedBytes(shape_) + 15) / 16 * 16 atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(pass.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        }
        else
        {
            const auto threads = pass.count * pass.lanes;
            [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        }
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    void Finish(id<MTLCommandBuffer> command)
    {
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError)
            throw std::runtime_error(command.error.localizedDescription.UTF8String);
    }
    std::vector<float> Read(id<MTLBuffer> source, std::size_t count)
    {
        std::vector<float> result(count);
        Stage(source, count * sizeof(float), nullptr, result.data());
        return result;
    }
    void Write(id<MTLBuffer> target, const std::vector<float>& source)
    {
        Stage(target, source.size() * sizeof(float), source.data(), nullptr);
    }
    // Copies host input into buffer, or buffer into host output, through a shared staging buffer.
    void Stage(id<MTLBuffer> buffer, std::size_t total, const void* input, void* output)
    {
        if (!total)
            return;
        @autoreleasepool
        {
            // Bounded staging avoids duplicating the full strategy buffer in unified memory.
            const auto capacity = std::min<std::size_t>(total, kReadbackBytes);
            id<MTLBuffer> staging = [device_ newBufferWithLength:capacity options:MTLResourceStorageModeShared];
            Check(staging, nil, "Cannot allocate Metal staging buffer");
            for (std::size_t offset = 0; offset < total; offset += capacity)
            {
                const auto bytes = std::min(capacity, total - offset);
                if (input)
                    std::memcpy(staging.contents, static_cast<const char*>(input) + offset, bytes);
                id<MTLCommandBuffer> command = [queue_ commandBuffer];
                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                if (input)
                    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:buffer destinationOffset:offset size:bytes];
                else
                    [blit copyFromBuffer:buffer sourceOffset:offset toBuffer:staging destinationOffset:0 size:bytes];
                [blit endEncoding];
                Finish(command);
                if (output)
                    std::memcpy(static_cast<char*>(output) + offset, staging.contents, bytes);
            }
        }
    }
};
} // namespace
bool DeviceAvailable()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device && [device supportsFamily:MTLGPUFamilyApple7];
    }
}
std::uint64_t DeviceMemoryBudget()
{
    @autoreleasepool
    {
        return MTLCreateSystemDefaultDevice().recommendedMaxWorkingSetSize;
    }
}
std::unique_ptr<Executor> MakeExecutor(const Plan& plan)
{
    return std::make_unique<MetalExecutor>(plan);
}
} // namespace solver::engine::gpu
