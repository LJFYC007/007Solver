// Trains an emulated-GPU session and prints a bitwise fingerprint of its training state
// and GPU-evaluated checkpoint. Race-free kernels give identical fingerprints for every
// dialect, thread order and legal lane schedule.
#include "engine/DcfrSession.h"
#include "game/GameCompiler.h"
#include "io/ScenarioLoader.h"
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace solver;
template<typename T>
std::uint64_t Hash(const std::vector<T>& values, std::size_t& nans)
{
    std::uint64_t h = 1469598103934665603ull;
    for (const auto& v : values)
    {
        if constexpr (std::is_floating_point_v<T>)
            nans += std::isnan(v) ? 1 : 0;
        unsigned char bytes[sizeof(T)];
        std::memcpy(bytes, &v, sizeof(T));
        for (auto b : bytes)
            h = (h ^ b) * 1099511628211ull;
    }
    return h;
}
int main(int argc, char** argv)
{
    auto scenario = io::LoadScenario(argv[1]);
    auto problem =
        std::make_shared<const engine::SolveProblem>(engine::SolveProblem{game::CompileGame(scenario.game), std::move(scenario.ranges)});
    engine::DcfrSession gpu(problem, engine::ComputeDevice::Gpu);
    gpu.Run(std::atoi(argv[2]));
    const auto state = gpu.ReadTrainingState();
    const auto metrics = gpu.EvaluateCheckpoint(false);
    std::size_t nans = 0;
    const auto r = Hash(state.regrets, nans), s = Hash(state.strategySums, nans), t = Hash(state.stamps, nans);
    std::uint32_t e;
    std::memcpy(&e, &metrics.exploitability, 4);
    std::printf(
        "%-40s regrets %016llx sums %016llx stamps %016llx gpuExpl %08x (%g) nans %zu\n",
        gpu.DeviceName(),
        (unsigned long long)r,
        (unsigned long long)s,
        (unsigned long long)t,
        e,
        metrics.exploitability,
        nans
    );
}
