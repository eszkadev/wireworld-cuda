#include "simulatorcuda.hpp"

extern "C" int CUDA_step(Model* pModel);

SimulatorCUDA::SimulatorCUDA(Model* pModel)
    : Simulator(pModel)
{

}

void SimulatorCUDA::Setup()
{
}

void SimulatorCUDA::Step()
{
    if(CUDA_step(m_pModel) != 0)
        throw;
}
