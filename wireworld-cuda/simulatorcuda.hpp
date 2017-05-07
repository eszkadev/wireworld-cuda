#ifndef SIMULATORCUDA_HPP
#define SIMULATORCUDA_HPP

#include <simulator.hpp>

class SimulatorCUDA : public Simulator
{
public:
    SimulatorCUDA(Model* pModel);

    virtual void Setup();
    virtual void Step();
};

#endif // SIMULATORCUDA_HPP
