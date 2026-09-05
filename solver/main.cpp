#include "service/SolverService.h"
#include <iostream>

int main(int argc, char* argv[])
{
    solver::service::SolverService service(std::cin, std::cout, std::cerr);
    if (argc != 2)
        return service.Fail("Expected the scenario file path as the only argument");
    return service.Run(argv[1]);
}
