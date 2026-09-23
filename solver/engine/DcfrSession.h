#pragma once

#include "engine/CpuDcfrSession.h"
#include "engine/GpuDcfrSession.h"
#include <functional>
#include <optional>

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
    // One iteration updates one player, alternating across successive runs.
    void Run(int iterations, const std::function<void(int)>& progressCallback = {});
    // Final metrics always use CPU evaluation. A provisional GPU checkpoint that
    // reaches stoppingTarget is CPU-certified before it can authorize stopping.
    ExploitabilityMetrics EvaluateCheckpoint(bool final, std::optional<double> stoppingTarget = std::nullopt) const;
    // Consumes training state; only metadata remains available afterward.
    StrategySnapshot ExportStrategy() &&;
    // Exact resident state in the layout both devices share, for lockstep parity checks.
    // It omits the iteration count that selects the player and discounts, so sessions
    // exchanging state must have completed equal iterations. Copies are outside Memory().
    TrainingState ReadTrainingState() const;
    void WriteTrainingState(const TrainingState& state);
    int CompletedIterations() const { return completedIterations_; }
    float TrainingTimeSeconds() const { return trainingTimeSeconds_; }
    int WorkerCount() const;
    ComputeDevice Device() const { return gpu_ ? ComputeDevice::Gpu : ComputeDevice::Cpu; }
    const char* DeviceName() const;
    const MemoryEstimate& Memory() const { return memory_; }

private:
    void CheckActive() const;
    std::unique_ptr<CpuDcfrSession> cpu_;
    std::unique_ptr<GpuDcfrSession> gpu_;
    MemoryEstimate memory_{};
    int completedIterations_ = 0;
    // Product of positive regret discounts over completed rounds (one update per player).
    double positiveScale_ = 1.0;
    float trainingTimeSeconds_ = 0.0f;
    bool exported_ = false;
};
} // namespace solver::engine
