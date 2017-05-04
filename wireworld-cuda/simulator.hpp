#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <model.hpp>

class Simulator
{
    Model* m_pModel;

public:
    Simulator(Model* pModel)
        : m_pModel(pModel)
    {
    }

    virtual void Setup() = 0;
    virtual void Step() = 0;
};

#endif // SIMULATOR_HPP
