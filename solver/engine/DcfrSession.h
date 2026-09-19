#pragma once

#include "engine/CpuDcfrSession.h"
#include "engine/GpuDcfrSession.h"

namespace solver::engine
{
enum class ComputeDevice
{
    Auto,
    Cpu,
    Gpu
};

// The service lifecycle is shared; the recursive CPU implementation stays independent.
class DcfrSession
{
public:
    explicit DcfrSession(std::shared_ptr<const SolveProblem> problem, ComputeDevice device = ComputeDevice::Auto, int workers = 0);
    void Run(int iterations, const std::function<void(int)>& progressCallback = {});
    ExploitabilityMetrics EvaluateExploitability() const;
    ExploitabilityMetrics CertifyExploitability() const;
    StrategySnapshot ExportStrategy() &&;
    int CompletedIterations() const;
    float TrainingTimeSeconds() const;
    int WorkerCount() const;
    ComputeDevice Device() const { return gpu_ ? ComputeDevice::Gpu : ComputeDevice::Cpu; }
    const char* DeviceName() const;
    const MemoryEstimate& Memory() const { return memory_; }

private:
    std::unique_ptr<CpuDcfrSession> cpu_;
    std::unique_ptr<GpuDcfrSession> gpu_;
    MemoryEstimate memory_{};
};
} // namespace solver::engine
