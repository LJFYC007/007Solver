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
    void Observe(int iterations, double exploitability, double trainingSeconds, double evaluationSeconds);
    std::optional<double> TargetIteration(double target) const;
    int NextCheck(int iterationLimit, double target) const;
    std::optional<double> RemainingSeconds(int completed, int nextCheck, int iterationLimit, double target) const;
    double FinalizationSeconds() const;

private:
    struct Sample
    {
        int iterations = 0;
        double exploitability = 0.0;
        double trainingSeconds = 0.0;
        double evaluationSeconds = 0.0;
    };
    double SecondsPerIteration() const;
    double CheckInterval() const;
    double MinimumInterval() const;
    std::array<Sample, 4> samples_{};
    std::size_t size_ = 0;
};
} // namespace solver::service
