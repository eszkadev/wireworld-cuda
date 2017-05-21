#ifndef SIMULATORCUDA_HPP
#define SIMULATORCUDA_HPP

#include <simulator.hpp>

class SimulatorCUDA : public Simulator
{
public:
    SimulatorCUDA(Model* pModel);
    ~SimulatorCUDA();

    virtual void Setup();
    virtual void Step(int n);
};

#endif // SIMULATORCUDA_HPP
