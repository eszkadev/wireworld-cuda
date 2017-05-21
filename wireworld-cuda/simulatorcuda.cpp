#include "simulatorcuda.hpp"

extern "C" int CUDA_step(Model* pModel, int n);
extern "C" void CUDA_setup();
extern "C" void CUDA_exit();

SimulatorCUDA::SimulatorCUDA(Model* pModel)
    : Simulator(pModel)
{
}

SimulatorCUDA::~SimulatorCUDA()
{
    CUDA_exit();
}

void SimulatorCUDA::Setup()
{
    CUDA_setup();
}

void SimulatorCUDA::Step(int n)
{
    if(CUDA_step(m_pModel, n) != 0)
        throw;
}
