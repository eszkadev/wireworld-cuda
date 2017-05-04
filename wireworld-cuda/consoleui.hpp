#ifndef CONSOLEUI_HPP
#define CONSOLEUI_HPP

#include <model.hpp>
#include <simulator.hpp>

class Consoleui
{
    Model* m_pModel;
    Simulator* m_pSimulator;

public:
    Consoleui();

    bool Run();
    void DrawMap();
};

#endif // CONSOLEUI_HPP
