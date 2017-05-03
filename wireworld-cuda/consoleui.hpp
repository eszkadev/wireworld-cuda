#ifndef CONSOLEUI_HPP
#define CONSOLEUI_HPP

#include <model.hpp>

class Consoleui
{
    Model* m_pModel;

public:
    Consoleui();

    bool Run();
    void DrawMap();
};

#endif // CONSOLEUI_HPP
