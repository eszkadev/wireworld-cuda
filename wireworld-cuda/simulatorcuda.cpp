#include "simulatorcuda.hpp"

extern "C" int CUDA_step(Model* pModel);
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

void SimulatorCUDA::Step()
{
    if(CUDA_step(m_pModel) != 0)
        throw;
}
