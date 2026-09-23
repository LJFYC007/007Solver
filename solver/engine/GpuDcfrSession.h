#pragma once

#include "engine/HandTraversal.h"
#include "engine/MemoryEstimate.h"
#include "engine/StrategyEvaluator.h"
#include "engine/gpu/GpuPlan.h"
#include <memory>

namespace solver::engine
{
class GpuDcfrSession
{
public:
    explicit GpuDcfrSession(const SolveProblem& problem);
    static bool Available();
    const char* DeviceName() const { return executor_->Name(); }
    const MemoryEstimate& Memory() const { return memory_; }

private:
    friend class DcfrSession;
    void Update(std::size_t player, float positiveDiscount, float averageDiscount);
    ExploitabilityMetrics EvaluateExploitability() const;
    // Independent CPU certification of the resident average policy.
    ExploitabilityMetrics EvaluateExploitabilityOnCpu() const;
    StrategySnapshot ExportStrategy() &&;
    TrainingState ReadTrainingState() const { return executor_->DownloadTraining(); }
    void WriteTrainingState(const TrainingState& state);
    std::shared_ptr<const HandTraversalData> data_;
    std::unique_ptr<gpu::Executor> executor_;
    gpu::State state_{};
    MemoryEstimate memory_{};
};
} // namespace solver::engine
