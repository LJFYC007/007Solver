// Replaces GpuUnavailable.cpp: a GPU "device" whose kernels are the real GpuKernels.inc
// executed on the host through the CUDA or Metal dialect.
//
// Environment:
//   SOLVER_GPU_EMULATION           cuda (default) | metal
//   SOLVER_GPU_EMULATION_ORDER     forward (default) | reverse   thread/block order in each pass
//   SOLVER_GPU_EMULATION_SCHEDULE  inorder (default) | lane1-late | lane0-late
//       Legal CUDA-graph orders of the two river lanes under Pass::sync; a missing sync
//       edge between lanes that share data changes results under one of them.
#include "Dialect.h"
#include "FiberRuntime.h"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

asm(R"(
    .pushsection .text
    .globl EmuSwitch
    .type EmuSwitch,@function
EmuSwitch:
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    subq $8, %rsp
    stmxcsr (%rsp)
    fnstcw 4(%rsp)
    movq %rsp, (%rdi)
    movq %rsi, %rsp
    ldmxcsr (%rsp)
    fldcw 4(%rsp)
    addq $8, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
    .size EmuSwitch,.-EmuSwitch
    .popsection
)");

extern "C" void EmuFiberEntry()
{
    emu::Group::Current().EnterFiber();
}

namespace solver::engine::gpu
{
namespace
{
std::string Env(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

enum class Schedule
{
    InOrder,
    Lane1Late,
    Lane0Late,
};

class EmulatedExecutor final : public Executor
{
public:
    explicit EmulatedExecutor(const Plan& plan) : passes_(plan.passes), shape_(plan.state), entries_(plan.entries)
    {
        const auto dialect = Env("SOLVER_GPU_EMULATION", "cuda");
        dialect_ = dialect == "metal" ? emu::MakeMetalDialect() : dialect == "cuda" ? emu::MakeCudaDialect() : nullptr;
        if (!dialect_)
            throw std::runtime_error("SOLVER_GPU_EMULATION must be cuda or metal");
        const auto order = Env("SOLVER_GPU_EMULATION_ORDER", "forward");
        if (order != "forward" && order != "reverse")
            throw std::runtime_error("SOLVER_GPU_EMULATION_ORDER must be forward or reverse");
        reverse_ = order == "reverse";
        const auto schedule = Env("SOLVER_GPU_EMULATION_SCHEDULE", "inorder");
        if (schedule == "inorder")
            schedule_ = Schedule::InOrder;
        else if (schedule == "lane1-late")
            schedule_ = Schedule::Lane1Late;
        else if (schedule == "lane0-late")
            schedule_ = Schedule::Lane0Late;
        else
            throw std::runtime_error("SOLVER_GPU_EMULATION_SCHEDULE must be inorder, lane1-late or lane0-late");
        name_ = std::string(dialect_->Name()) + " emulation (" + order + ", " + schedule + ")";

        const auto sources = plan.Buffers();
        for (std::size_t i = 0; i < kBufferCount; ++i)
        {
            auto& buffer = buffers_.data[i];
            buffer.resize(sources[i].AllocationBytes());
            if (sources[i].data)
                std::memcpy(buffer.data(), sources[i].data, sources[i].bytes);
            else
                Poison(i);
        }
        // Both executors zero exactly these device-only buffers.
        for (const auto index : {RegretsBuffer, SumsBuffer, FlagsBuffer, StampsBuffer})
            std::memset(buffers_.data[index].data(), 0, sources[index].bytes);
        SetState(shape_);
        for (const auto& pass : plan.initialization)
            Execute(pass, 0);
    }
    const char* Name() const override { return name_.c_str(); }
    void Update(const State& state) override
    {
        SetState(state);
        std::vector<Pass> deferred;
        std::size_t forked = 0; // deferred lane-0 passes that the latest fork covers
        for (const auto& pass : passes_)
            switch (schedule_)
            {
            case Schedule::InOrder:
                Execute(pass, state.player);
                break;
            case Schedule::Lane1Late:
                // Lane 1 runs as late as possible: only kJoinBefore (and the end) waits for it.
                if (pass.lane)
                    deferred.push_back(pass);
                else
                {
                    if (pass.sync & kJoinBefore)
                        Flush(deferred, deferred.size(), state.player);
                    Execute(pass, state.player);
                }
                break;
            case Schedule::Lane0Late:
                // Lane 0 runs as late as possible: only a kWaitFork lane-1 pass waits for it,
                // and only up to the latest kForkAfter pass.
                if (!pass.lane)
                {
                    deferred.push_back(pass);
                    if (pass.sync & kForkAfter)
                        forked = deferred.size();
                }
                else
                {
                    if (pass.sync & kWaitFork)
                    {
                        Flush(deferred, forked, state.player);
                        forked = 0;
                    }
                    Execute(pass, state.player);
                }
                break;
            }
        Flush(deferred, deferred.size(), state.player);
    }
    std::vector<float> RootValues(const State& state) override
    {
        Update(state);
        const auto* values = buffers_.As<const float>(ValuesBuffer);
        return std::vector<float>(values, values + state.hands[state.player]);
    }
    std::vector<float> DownloadSums(bool releaseTraining) override
    {
        if (releaseTraining)
            for (std::size_t i = 0; i < kBufferCount; ++i)
                if (i != SumsBuffer)
                    std::vector<unsigned char>().swap(buffers_.data[i]);
        const auto* sums = buffers_.As<const float>(SumsBuffer);
        std::vector<float> result(sums, sums + entries_);
        if (releaseTraining)
            std::vector<unsigned char>().swap(buffers_.data[SumsBuffer]);
        return result;
    }
    TrainingState DownloadTraining() override
    {
        const auto* regrets = buffers_.As<const float>(RegretsBuffer);
        const auto* sums = buffers_.As<const float>(SumsBuffer);
        const auto* stamps = buffers_.As<const std::uint32_t>(StampsBuffer);
        TrainingState state{std::vector<float>(regrets, regrets + entries_), std::vector<float>(sums, sums + entries_)};
        state.stamps = NewestStamps(std::vector<std::uint32_t>(stamps, stamps + 2 * std::size_t(shape_.stampCount)));
        return state;
    }
    void UploadTraining(const TrainingState& state) override
    {
        std::memcpy(buffers_.data[RegretsBuffer].data(), state.regrets.data(), entries_ * sizeof(float));
        std::memcpy(buffers_.data[SumsBuffer].data(), state.strategySums.data(), entries_ * sizeof(float));
        for (std::size_t half = 0; half < 2; ++half)
            std::memcpy(
                buffers_.As<std::uint32_t>(StampsBuffer) + half * shape_.stampCount,
                state.stamps.data(),
                shape_.stampCount * sizeof(std::uint32_t)
            );
    }

private:
    std::vector<Pass> passes_;
    State shape_;
    std::size_t entries_;
    emu::DeviceBuffers buffers_;
    std::unique_ptr<emu::Dialect> dialect_;
    bool reverse_ = false;
    Schedule schedule_ = Schedule::InOrder;
    std::string name_;

    void Poison(std::size_t index)
    {
        auto& buffer = buffers_.data[index];
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t offset = 0; offset + sizeof(float) <= buffer.size(); offset += sizeof(float))
            std::memcpy(buffer.data() + offset, &nan, sizeof(float));
    }
    void SetState(const State& state) { std::memcpy(buffers_.data[StateBuffer].data(), &state, sizeof(State)); }
    void Execute(const Pass& pass, U32 player) { dialect_->Dispatch(buffers_, LaunchPass(pass, shape_, player), player, shape_, reverse_); }
    void Flush(std::vector<Pass>& deferred, std::size_t count, U32 player)
    {
        for (std::size_t i = 0; i < count; ++i)
            Execute(deferred[i], player);
        deferred.erase(deferred.begin(), deferred.begin() + static_cast<std::ptrdiff_t>(count));
    }
};
} // namespace

bool DeviceAvailable()
{
    return Env("SOLVER_GPU_EMULATION", "cuda") != "off";
}
std::uint64_t DeviceMemoryBudget()
{
    return std::stoull(Env("SOLVER_GPU_EMULATION_BUDGET", "8589934592"));
}
std::unique_ptr<Executor> MakeExecutor(const Plan& plan)
{
    return std::make_unique<EmulatedExecutor>(plan);
}
} // namespace solver::engine::gpu
