#ifndef SIMULATORCPP_HPP
#define SIMULATORCPP_HPP

#include <simulator.hpp>

class SimulatorCPP : public Simulator
{
public:
    SimulatorCPP(Model* pModel);

    virtual void Setup();
    virtual void Step();
};

#endif // SIMULATORCPP_HPP
