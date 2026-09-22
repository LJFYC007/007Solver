#include "engine/GpuDcfrSession.h"
#include <algorithm>
#include <stdexcept>

namespace solver::engine
{
GpuDcfrSession::GpuDcfrSession(std::shared_ptr<const SolveProblem> problem) : problem_(std::move(problem))
{
    if (!problem_ || !problem_->game)
        throw std::invalid_argument("GPU DCFR session requires a solve problem");
    if (!Available())
        throw std::runtime_error("A supported CUDA or Apple Silicon GPU is unavailable");
    memory_ = EstimateCpuMemory(*problem_, 1);
    if (memory_.strategyEntries > gpu::DeviceMemoryBudget() / (2 * sizeof(float)))
        throw std::runtime_error("Regret and cumulative strategy alone exceed GPU memory; reduce the tree or use CPU");
    data_ = std::make_shared<const HandTraversalData>(*problem_, problem_->game->Root());
    const gpu::Plan plan(*data_);
    state_ = plan.state;
    memory_.workers = 0;
    const auto fixed = HandTraversal::EstimateStorage(*problem_->game, {data_->tables->hands[0].size(), data_->tables->hands[1].size()});
    // Combined host/device allocation estimate, not a claim about available VRAM.
    const auto host = problem_->game->Size().storageBytes + fixed.fixedBytes;
    const auto sumsBytes = plan.Buffers()[gpu::SumsBuffer].bytes;
    const auto certification = sumsBytes + fixed.workspaceBytes;
    const auto size = problem_->game->Size();
    const auto snapshot =
        StrategySnapshot::EstimateStorageBytes(size.decisionNodes[0] + size.decisionNodes[1], data_->infoSetCount, plan.entries);
    const auto exportPeak = host + sumsBytes + snapshot + gpu::kReadbackBytes;
    const auto peak = std::max(host + plan.DeviceBytes() + std::max(plan.HostBytes(), certification), exportPeak);
    memory_.peakBytes = peak + peak / 8 + 64 * 1024 * 1024;
    executor_ = gpu::MakeExecutor(plan);
}

bool GpuDcfrSession::Available()
{
    return gpu::DeviceAvailable();
}

void GpuDcfrSession::Update(std::size_t player, float positiveDiscount, float averageDiscount)
{
    auto state = state_;
    state.player = static_cast<gpu::U32>(player);
    state.positiveDiscount = positiveDiscount;
    state.averageDiscount = averageDiscount;
    executor_->Update(state);
}

ExploitabilityMetrics GpuDcfrSession::EvaluateExploitability() const
{
    std::array<float, 2> response{};
    auto state = state_;
    state.evaluation = 1;
    for (gpu::U32 p = 0; p < 2; ++p)
    {
        state.player = p;
        const auto values = executor_->RootValues(state);
        float value = 0.0f, mass = 0.0f;
        for (std::size_t h = 0; h < values.size(); ++h)
        {
            const auto& hand = data_->tables->hands[p][h];
            const float weight = hand.weight * hand.opponentMass;
            value += weight * values[h];
            mass += weight;
        }
        response[p] = value / mass;
    }
    return {response[0], response[1], (response[0] + response[1]) / 2.0f};
}

ExploitabilityMetrics GpuDcfrSession::EvaluateExploitabilityOnCpu() const
{
    const auto sums = executor_->DownloadSums(false);
    return EvaluateAverageStrategy(HandTraversal(data_), sums.data());
}

StrategySnapshot GpuDcfrSession::ExportStrategy() &&
{
    auto sums = executor_->DownloadSums(true);
    return data_->ExportStrategy(std::move(sums));
}
} // namespace solver::engine
