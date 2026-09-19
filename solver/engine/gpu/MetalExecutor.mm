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
            const char* names[] = {"Reach", "Prefix", "Terminal", "Backup", "Outcomes"};
            for (std::size_t i = 0; i < pipelines_.size(); ++i)
            {
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
                Check(function, nil, "Missing DCFR Metal kernel");
                pipelines_[i] = [device_ newComputePipelineStateWithFunction:function error:&error];
                Check(pipelines_[i], error, "Cannot create DCFR Metal pipeline");
            }
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
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            [blit fillBuffer:buffers_[9] range:NSMakeRange(0, buffers_[9].length) value:0];
            [blit fillBuffer:buffers_[10] range:NSMakeRange(0, buffers_[10].length) value:0];
            [blit endEncoding];
            if (shape_.outcomeRows)
            {
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                Bind(encoder, shape_);
                Encode(encoder, {Kernel::Outcomes, 0, static_cast<U32>(plan.outcomeEntries)});
                if (shape_.outcomeRows > 1)
                    Encode(encoder, {Kernel::Outcomes, 1, shape_.hands[0] * shape_.hands[1]});
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
        return Read(buffers_[12], state.hands[state.player]);
    }
    std::vector<float> DownloadSums(bool releaseTraining) override
    {
        if (releaseTraining)
            for (std::size_t i = 0; i < buffers_.size(); ++i)
                if (i != 10)
                    buffers_[i] = nil;
        auto result = Read(buffers_[10], entries_);
        if (releaseTraining)
            buffers_[10] = nil;
        return result;
    }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    std::array<id<MTLBuffer>, 13> buffers_{};
    std::array<id<MTLComputePipelineState>, 5> pipelines_{};
    std::vector<Pass> passes_;
    State shape_;
    std::size_t entries_;
    void Allocate(std::size_t i, std::size_t bytes)
    {
        if (bytes > device_.maxBufferLength)
            throw std::runtime_error("A DCFR buffer exceeds Metal maxBufferLength");
        buffers_[i] = [device_ newBufferWithLength:std::max<std::size_t>(bytes, 4) options:MTLResourceStorageModePrivate];
        Check(buffers_[i], nil, "Cannot allocate DCFR Metal buffer");
    }
    template<typename T>
    void Upload(std::size_t i, const std::vector<T>& data)
    {
        if (data.empty())
        {
            Allocate(i, 4);
            return;
        }
        buffers_[i] = [device_ newBufferWithBytes:data.data() length:data.size() * sizeof(T) options:MTLResourceStorageModeShared];
        Check(buffers_[i], nil, "Cannot allocate DCFR Metal table");
    }
    void Bind(id<MTLComputeCommandEncoder> encoder, const State& state)
    {
        for (std::size_t i = 0; i < buffers_.size(); ++i)
            [encoder setBuffer:buffers_[i] offset:0 atIndex:i];
        [encoder setBytes:&state length:sizeof(state) atIndex:13];
    }
    void Encode(id<MTLComputeCommandEncoder> encoder, Pass pass)
    {
        id<MTLComputePipelineState> pipeline = pipelines_[static_cast<std::size_t>(pass.operation)];
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&pass length:sizeof(pass) atIndex:14];
        const auto threads = pass.count * (pass.operation == Kernel::Prefix || pass.operation == Kernel::Outcomes ? 1 : shape_.stride);
        [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
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
        if (!count)
            return result;
        @autoreleasepool
        {
            // Bounded staging avoids duplicating the full strategy buffer in unified memory.
            const auto capacity = std::min<std::size_t>(count * sizeof(float), 16 * 1024 * 1024);
            id<MTLBuffer> staging = [device_ newBufferWithLength:capacity options:MTLResourceStorageModeShared];
            Check(staging, nil, "Cannot allocate Metal readback buffer");
            for (std::size_t offset = 0; offset < count * sizeof(float); offset += capacity)
            {
                const auto bytes = std::min(capacity, count * sizeof(float) - offset);
                id<MTLCommandBuffer> command = [queue_ commandBuffer];
                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                [blit copyFromBuffer:source sourceOffset:offset toBuffer:staging destinationOffset:0 size:bytes];
                [blit endEncoding];
                Finish(command);
                std::memcpy(reinterpret_cast<char*>(result.data()) + offset, staging.contents, bytes);
            }
        }
        return result;
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
