#include "simulatorcuda.hpp"

extern "C" int CUDA_step(Model* pModel, int n);
extern "C" void CUDA_setup();
extern "C" void CUDA_exit();
extern "C" void CUDA_set(int nCells, int nBlock, int nGrid, int* pGpu);
extern "C" DevicesInfo* CUDA_getDevicesList();

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

void SimulatorCUDA::ApplySettings(int nCells, int nBlock, int nGrid, int* pGPU)
{
    CUDA_set(nCells, nBlock, nGrid, pGPU);
}

DevicesInfo* SimulatorCUDA::GetDeivces()
{
    return CUDA_getDevicesList();
}
