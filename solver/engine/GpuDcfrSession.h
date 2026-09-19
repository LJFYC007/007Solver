#pragma once

#include "engine/HandTraversal.h"
#include "engine/MemoryEstimate.h"
#include "engine/StrategyEvaluator.h"
#include "engine/gpu/GpuPlan.h"
#include <functional>
#include <memory>

namespace solver::engine
{
class GpuDcfrSession
{
public:
    explicit GpuDcfrSession(std::shared_ptr<const SolveProblem> problem);
    static bool Available();
    void Run(int iterations, const std::function<void(int)>& progressCallback = {});
    ExploitabilityMetrics EvaluateExploitability() const;
    // Independent CPU certification of the resident average policy.
    ExploitabilityMetrics EvaluateExploitabilityOnCpu() const;
    StrategySnapshot ExportStrategy() &&;
    int CompletedIterations() const { return completedIterations_; }
    float TrainingTimeSeconds() const { return trainingTimeSeconds_; }
    const char* DeviceName() const { return executor_->Name(); }
    const MemoryEstimate& Memory() const { return memory_; }

private:
    std::shared_ptr<const SolveProblem> problem_;
    std::shared_ptr<const HandTraversalData> data_;
    std::unique_ptr<gpu::Executor> executor_;
    gpu::State state_{};
    MemoryEstimate memory_{};
    int completedIterations_ = 0;
    float trainingTimeSeconds_ = 0.0f;
    bool exported_ = false;
};
} // namespace solver::engine
