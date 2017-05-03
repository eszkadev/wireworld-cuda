#include <consoleui.hpp>
#include <model.hpp>
#include <iostream>

using namespace std;

char lcl_CellToChar(Cell eCell)
{
    switch(eCell)
    {
    case Empty:
        return ' ';
    case Head:
        return 'H';
    case Tail:
        return 'T';
    case Conductor:
        return '+';
    default:
        return '.';
    }
}

Consoleui::Consoleui()
    : m_pModel(NULL)
{
}

bool Consoleui::Run()
{
    cout << "Wireworld CUDA by Szymon Kłos\n";
    cout << "-----------------------------\n";

    // TODO: event loop
    DrawMap();

    return true;
}

void Consoleui::DrawMap()
{
    if(m_pModel)
    {
        for(unsigned int y = 0; y < m_pModel->GetHeight(); ++y)
            for(unsigned int x = 0; x < m_pModel->GetWidth(); ++x)
                cout << lcl_CellToChar(m_pModel->GetCell(x, y));
    }
}
