#include "simulatorcuda.hpp"

extern "C" int CUDA_step(void);

SimulatorCUDA::SimulatorCUDA(Model* pModel)
    : Simulator(pModel)
{

}

void SimulatorCUDA::Setup()
{
}

void SimulatorCUDA::Step()
{
    if(CUDA_step() != 0)
        throw;
}
