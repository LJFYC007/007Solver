// CPU training must not depend on the worker count: parallel subtree updates keep the tree and
// action backup order (solver/ARCHITECTURE.md, "Training and memory"). Trains one session per
// worker count and requires bitwise-identical training state and final metrics.
// Usage: daily_cpu_determinism <scenario.json> <updates> <workers>...
#include "engine/DcfrSession.h"
#include "game/GameCompiler.h"
#include "io/ScenarioLoader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace solver;

namespace
{
template<typename T>
void Mix(std::uint64_t& hash, const std::vector<T>& values)
{
    for (const auto& value : values)
    {
        unsigned char bytes[sizeof(T)];
        std::memcpy(bytes, &value, sizeof(T));
        for (auto byte : bytes)
            hash = (hash ^ byte) * 1099511628211ull;
    }
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "Usage: %s <scenario.json> <updates> <workers>...\n", argv[0]);
        return 2;
    }
    auto scenario = io::LoadScenario(argv[1]);
    const auto problem =
        std::make_shared<const engine::SolveProblem>(engine::SolveProblem{game::CompileGame(scenario.game), std::move(scenario.ranges)});
    const int updates = std::atoi(argv[2]);
    std::string reference;
    int failures = 0;
    for (int i = 3; i < argc; ++i)
    {
        engine::DcfrSession session(problem, engine::ComputeDevice::Cpu, std::atoi(argv[i]));
        session.Run(updates);
        const auto state = session.ReadTrainingState();
        const auto metrics = session.EvaluateCheckpoint(true);
        std::uint64_t hash = 1469598103934665603ull;
        Mix(hash, state.regrets);
        Mix(hash, state.strategySums);
        Mix(hash, state.stamps);
        Mix(hash, std::vector<float>{metrics.player0BestResponseEv, metrics.player1BestResponseEv, metrics.exploitability});
        char line[160];
        std::snprintf(line, sizeof(line), "%016llx exploitability %.9g", static_cast<unsigned long long>(hash), metrics.exploitability);
        std::printf("workers %d: %s\n", session.WorkerCount(), line);
        if (reference.empty())
            reference = line;
        else if (reference != line)
            ++failures;
    }
    std::printf(failures ? "FAIL: training state depends on the worker count\n" : "PASS: identical for every worker count\n");
    return failures ? 1 : 0;
}
