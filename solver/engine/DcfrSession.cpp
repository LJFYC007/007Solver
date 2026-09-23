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
        gpu_ = std::make_unique<GpuDcfrSession>(*problem);
        memory_ = gpu_->Memory();
    }
    else
    {
        memory_ = EstimateCpuMemory(*problem, workers);
        cpu_ = std::make_unique<CpuDcfrSession>(*problem, workers);
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
        const auto t = static_cast<std::uint32_t>(completedIterations_ / 2 + 1);
        // Both discounts apply lazily: positive regrets are stored divided by the product
        // of their t^1.5 / (t^1.5 + 1) discounts, and cumulative strategies hold the sum of
        // t^2 * reach * policy, DCFR's (t / (t + 1))^2 discount rescaled away by normalization.
        const UpdateWeights weights{
            t, static_cast<float>(positiveScale_), static_cast<float>(1.0 / positiveScale_), static_cast<float>(t) * static_cast<float>(t)
        };
        if (gpu_)
            gpu_->Update(player, weights);
        else
            cpu_->Update(player, weights);
        ++completedIterations_;
        if (completedIterations_ % 2 == 0)
        {
            const double power = t * std::sqrt(static_cast<double>(t));
            positiveScale_ *= power / (power + 1.0);
        }
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
TrainingState DcfrSession::ReadTrainingState() const
{
    CheckActive();
    return gpu_ ? gpu_->ReadTrainingState() : cpu_->ReadTrainingState();
}
void DcfrSession::WriteTrainingState(const TrainingState& state)
{
    CheckActive();
    if (gpu_)
        gpu_->WriteTrainingState(state);
    else
        cpu_->WriteTrainingState(state);
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
