#include "service/MemoryBudget.h"
#include <algorithm>
#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#else
#include <mach/mach.h>
#endif

namespace solver::service
{
std::uint64_t AvailableSolveMemory()
{
    std::uint64_t available = 4ull * 1024 * 1024 * 1024;
#if defined(_WIN32)
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory))
        available = std::min(memory.ullAvailPhys, memory.ullAvailPageFile);
#else
    const mach_port_t host = mach_host_self();
    vm_size_t pageSize = 0;
    vm_statistics64_data_t memory{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_page_size(host, &pageSize) == KERN_SUCCESS &&
        host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&memory), &count) == KERN_SUCCESS)
        available = (static_cast<std::uint64_t>(memory.free_count) + memory.inactive_count) * pageSize;
    mach_port_deallocate(mach_task_self(), host);
#endif
    return available - available / 4;
}
} // namespace solver::service
