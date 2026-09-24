#pragma once
// Linux stand-in for the two Mach calls tests/benchmark.cpp uses: phys_footprint reports
// the current resident set from /proc (not equivalent to the macOS metric).
#include <fstream>
#include <string>
using kern_return_t = int;
using mach_msg_type_number_t = unsigned;
using task_info_t = int*;
struct task_vm_info_data_t
{
    unsigned long long phys_footprint = 0;
};
constexpr kern_return_t KERN_SUCCESS = 0;
constexpr int TASK_VM_INFO = 22;
constexpr mach_msg_type_number_t TASK_VM_INFO_COUNT = 1;
inline int mach_task_self()
{
    return 0;
}
inline kern_return_t task_info(int, int, task_info_t info, mach_msg_type_number_t*)
{
    std::ifstream status("/proc/self/status");
    for (std::string key; status >> key;)
        if (key == "VmRSS:")
        {
            unsigned long long kilobytes = 0;
            status >> kilobytes;
            reinterpret_cast<task_vm_info_data_t*>(info)->phys_footprint = kilobytes * 1024;
            return KERN_SUCCESS;
        }
    return 1;
}
