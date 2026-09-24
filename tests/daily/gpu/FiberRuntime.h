#pragma once
// Cooperative SIMT emulation of one thread block / threadgroup on the host CPU.
// Each GPU thread is a fiber with its own stack; block barriers and warp collectives
// suspend the fiber until every live thread of the block (or warp) arrives.
// Divergent barriers, collectives with exited lanes and deadlocks abort loudly.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>
#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>
#endif

extern "C" void EmuSwitch(void** save, void* load);
extern "C" void EmuFiberEntry();

namespace emu
{
constexpr unsigned kWarp = 32;
constexpr std::size_t kStackBytes = 128 * 1024;

[[noreturn]] inline void Fail(const char* message)
{
    std::fprintf(stderr, "GPU EMULATION FAILURE: %s\n", message);
    std::fflush(stderr);
    std::abort();
}

struct Fiber
{
    void* sp = nullptr;
    std::unique_ptr<unsigned char[]> stack;
    unsigned local = 0;
    bool done = false;
    const unsigned* wait = nullptr;
    unsigned waitValue = 0;
};

class Group
{
public:
    static Group*& Active()
    {
        thread_local Group* active = nullptr;
        return active;
    }
    static Group& Current()
    {
        auto* group = Active();
        if (!group)
            Fail("block collective called outside a cooperative group");
        return *group;
    }
    unsigned Local() const { return current_->local; }

    // Runs body(local) for every local in [0, size); reverse resumes high locals first.
    void Run(unsigned size, bool reverse, const std::function<void(unsigned)>& body)
    {
        if (fibers_.size() < size)
        {
            fibers_.resize(size);
            for (auto& fiber : fibers_)
                if (!fiber.stack)
                    fiber.stack.reset(new unsigned char[kStackBytes]);
        }
        const unsigned warps = (size + kWarp - 1) / kWarp;
        size_ = size;
        live_ = size;
        arrived_ = 0;
        generation_ = 0;
        warpArrived_.assign(warps, 0);
        warpGeneration_.assign(warps, 0);
        warpLive_.assign(warps, 0);
        exchange_.assign(2 * std::size_t(warps) * kWarp, 0.0f);
        votes_.assign(size, 0);
        for (unsigned i = 0; i < size; ++i)
        {
            auto& fiber = fibers_[i];
            fiber.local = i;
            fiber.done = false;
            fiber.wait = nullptr;
#if defined(__SANITIZE_ADDRESS__)
            // Frames abandoned by the previous block's fibers leave redzones poisoned.
            ASAN_UNPOISON_MEMORY_REGION(fiber.stack.get(), kStackBytes);
#endif
            fiber.sp = InitialStack(fiber);
            ++warpLive_[i / kWarp];
        }
        body_ = &body;
        auto* previous = Active();
        Active() = this;
        unsigned remaining = size;
        while (remaining)
        {
            bool progress = false;
            for (unsigned k = 0; k < size; ++k)
            {
                auto& fiber = fibers_[reverse ? size - 1 - k : k];
                if (fiber.done || (fiber.wait && *fiber.wait == fiber.waitValue))
                    continue;
                fiber.wait = nullptr;
                current_ = &fiber;
                progress = true;
                EmuSwitch(&scheduler_, fiber.sp);
                if (fiber.done)
                    --remaining;
            }
            if (!progress)
                Fail("deadlock: threads wait at different barriers or collectives");
        }
        Active() = previous;
        current_ = nullptr;
    }

    // __syncthreads / threadgroup_barrier; the vote implements __syncthreads_or.
    bool BarrierOr(bool vote)
    {
        const unsigned generation = generation_;
        votes_[current_->local] = vote ? 1 : 0;
        if (++arrived_ == live_)
            ReleaseBarrier();
        else
            Wait(&generation_, generation);
        return result_[generation & 1];
    }
    void Barrier() { BarrierOr(false); }

    // Every lane of the caller's warp deposits a value; returns the warp's 32 values.
    // All 32 lanes must be live, as full-mask CUDA shuffles and Metal SIMD ops require.
    const float* WarpCollect(float value)
    {
        const unsigned local = current_->local, warp = local / kWarp;
        if (warpLive_[warp] != kWarp)
            Fail("warp collective with inactive lanes");
        const unsigned generation = warpGeneration_[warp];
        float* slot = exchange_.data() + (std::size_t(generation & 1) * warpLive_.size() + warp) * kWarp;
        slot[local % kWarp] = value;
        if (++warpArrived_[warp] == kWarp)
        {
            warpArrived_[warp] = 0;
            ++warpGeneration_[warp];
        }
        else
            Wait(&warpGeneration_[warp], generation);
        return slot;
    }

    void EnterFiber()
    {
        auto& fiber = *current_;
        (*body_)(fiber.local);
        fiber.done = true;
        --live_;
        const unsigned warp = fiber.local / kWarp;
        --warpLive_[warp];
        if (warpArrived_[warp])
            Fail("a lane exited while its warp waits in a collective");
        if (arrived_ && arrived_ == live_)
            ReleaseBarrier();
        EmuSwitch(&fiber.sp, scheduler_);
        Fail("finished fiber resumed");
    }

private:
    std::vector<Fiber> fibers_;
    Fiber* current_ = nullptr;
    void* scheduler_ = nullptr;
    const std::function<void(unsigned)>* body_ = nullptr;
    unsigned size_ = 0, live_ = 0, arrived_ = 0, generation_ = 0;
    bool result_[2] = {false, false};
    std::vector<unsigned> warpArrived_, warpGeneration_, warpLive_;
    std::vector<float> exchange_;
    std::vector<unsigned char> votes_;

    void Wait(const unsigned* generation, unsigned value)
    {
        auto& fiber = *current_;
        fiber.wait = generation;
        fiber.waitValue = value;
        EmuSwitch(&fiber.sp, scheduler_);
    }
    void ReleaseBarrier()
    {
        bool any = false;
        for (unsigned i = 0; i < size_; ++i)
            if (!fibers_[i].done)
                any |= votes_[i] != 0;
        result_[generation_ & 1] = any;
        arrived_ = 0;
        ++generation_;
    }
    static void* InitialStack(Fiber& fiber)
    {
        // EmuSwitch pops MXCSR/x87 CW, r15..r12, rbx, rbp, then returns into EmuFiberEntry
        // with rsp == 8 (mod 16), as after a call.
        const auto top = (reinterpret_cast<std::uintptr_t>(fiber.stack.get()) + kStackBytes) & ~std::uintptr_t(15);
        auto* words = reinterpret_cast<std::uint64_t*>(top);
        words[-1] = 0;
        words[-2] = reinterpret_cast<std::uint64_t>(&EmuFiberEntry);
        for (int i = 3; i <= 8; ++i)
            words[-i] = 0;
        std::uint32_t mxcsr;
        std::uint16_t fpucw;
        asm volatile("stmxcsr %0" : "=m"(mxcsr));
        asm volatile("fnstcw %0" : "=m"(fpucw));
        auto* csr = reinterpret_cast<unsigned char*>(top - 72);
        *reinterpret_cast<std::uint64_t*>(csr) = 0;
        *reinterpret_cast<std::uint32_t*>(csr) = mxcsr;
        *reinterpret_cast<std::uint16_t*>(csr + 4) = fpucw;
        return csr;
    }
};
} // namespace emu
