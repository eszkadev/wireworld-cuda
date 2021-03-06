#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <model.hpp>

class Simulator
{
protected:
    Model* m_pModel;

public:
    Simulator(Model* pModel)
        : m_pModel(pModel)
    {
    }

    virtual ~Simulator() {}

    virtual void Setup() = 0;
    virtual void Step(int n) = 0;
};

#endif // SIMULATOR_HPP
