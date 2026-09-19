#pragma once

#include <array>
#include <cstddef>
#include <optional>

namespace solver::service
{
// Estimates when a measured average strategy will meet a target. It never stops training.
class ConvergenceEstimate
{
public:
    void Observe(int iterations, float exploitability, float trainingSeconds, float evaluationSeconds);
    std::optional<float> TargetIteration(float target) const;
    int NextCheck(int iterationLimit, float target) const;
    std::optional<float> RemainingSeconds(int completed, int nextCheck, int iterationLimit, float target) const;
    float FinalizationSeconds() const;

private:
    struct Sample
    {
        int iterations = 0;
        float exploitability = 0.0f;
        float trainingSeconds = 0.0f;
        float evaluationSeconds = 0.0f;
    };
    float SecondsPerIteration() const;
    float CheckInterval() const;
    float MinimumInterval() const;
    std::array<Sample, 4> samples_{};
    std::size_t size_ = 0;
};
} // namespace solver::service
