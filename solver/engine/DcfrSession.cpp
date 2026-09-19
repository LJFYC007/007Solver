#include "engine/DcfrSession.h"
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
    if (gpu_)
        gpu_->Run(iterations, callback);
    else
        cpu_->Run(iterations, callback);
}
ExploitabilityMetrics DcfrSession::EvaluateExploitability() const
{
    return gpu_ ? gpu_->EvaluateExploitability() : cpu_->EvaluateExploitability();
}
ExploitabilityMetrics DcfrSession::CertifyExploitability() const
{
    return gpu_ ? gpu_->EvaluateExploitabilityOnCpu() : cpu_->EvaluateExploitability();
}
StrategySnapshot DcfrSession::ExportStrategy() &&
{
    return gpu_ ? std::move(*gpu_).ExportStrategy() : std::move(*cpu_).ExportStrategy();
}
int DcfrSession::CompletedIterations() const
{
    return gpu_ ? gpu_->CompletedIterations() : cpu_->CompletedIterations();
}
float DcfrSession::TrainingTimeSeconds() const
{
    return gpu_ ? gpu_->TrainingTimeSeconds() : cpu_->TrainingTimeSeconds();
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
