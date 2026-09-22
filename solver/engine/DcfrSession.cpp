#include "engine/DcfrSession.h"
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace solver::engine
{
DcfrSession::DcfrSession(std::shared_ptr<const SolveProblem> problem, ComputeDevice device, int workers)
{
    if (!problem || !problem->game)
        throw std::invalid_argument("DCFR session requires a solve problem");
    if (device == ComputeDevice::Gpu || (device == ComputeDevice::Auto && GpuDcfrSession::Available()))
    {
        gpu_ = std::make_unique<GpuDcfrSession>(std::move(problem));
        memory_ = gpu_->Memory();
    }
    else
    {
        memory_ = EstimateCpuMemory(*problem, workers);
        cpu_ = std::make_unique<CpuDcfrSession>(std::move(problem), workers);
    }
}
void DcfrSession::Run(int iterations, const std::function<void(int)>& callback)
{
    CheckActive();
    if (iterations <= 0 || iterations > std::numeric_limits<int>::max() - completedIterations_)
        throw std::invalid_argument("DCFR iterations must be positive and fit the completed iteration counter");
    const auto start = std::chrono::steady_clock::now();
    auto lastProgress = start;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const auto player = static_cast<std::size_t>(completedIterations_ % 2);
        const float t = completedIterations_ / 2 + 1.0f;
        const float power = t * std::sqrt(t);
        const float positiveDiscount = power / (power + 1.0f);
        const float averageDiscount = (t / (t + 1.0f)) * (t / (t + 1.0f));
        if (gpu_)
            gpu_->Update(player, positiveDiscount, averageDiscount);
        else
            cpu_->Update(player, positiveDiscount, averageDiscount);
        ++completedIterations_;
        if (callback)
        {
            const auto now = std::chrono::steady_clock::now();
            if (iteration + 1 == iterations || now - lastProgress >= std::chrono::milliseconds(250))
            {
                callback(completedIterations_);
                lastProgress = now;
            }
        }
    }
    trainingTimeSeconds_ += std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
}
ExploitabilityMetrics DcfrSession::EvaluateCheckpoint(bool final, std::optional<double> stoppingTarget) const
{
    CheckActive();
    if (!gpu_)
        return cpu_->EvaluateExploitability();
    if (final)
        return gpu_->EvaluateExploitabilityOnCpu();
    const auto metrics = gpu_->EvaluateExploitability();
    return stoppingTarget && metrics.exploitability <= *stoppingTarget ? gpu_->EvaluateExploitabilityOnCpu() : metrics;
}
StrategySnapshot DcfrSession::ExportStrategy() &&
{
    CheckActive();
    exported_ = true;
    return gpu_ ? std::move(*gpu_).ExportStrategy() : std::move(*cpu_).ExportStrategy();
}
void DcfrSession::CheckActive() const
{
    if (exported_)
        throw std::logic_error("DCFR training state has been exported");
}
int DcfrSession::WorkerCount() const
{
    return gpu_ ? 0 : cpu_->WorkerCount();
}
const char* DcfrSession::DeviceName() const
{
    return gpu_ ? gpu_->DeviceName() : "CPU";
}
} // namespace solver::engine
