#include <consoleui.hpp>
#include <model.hpp>
#include <iostream>

using namespace std;

Consoleui::Consoleui()
    : m_pModel(NULL)
{
}

bool Consoleui::Run()
{
    cout << "Wireworld CUDA by Szymon Kłos\n";
    cout << "-----------------------------\n";

    m_pModel = new Model();
    m_pModel->LoadModel("model1.txt");

    // TODO: event loop
    DrawMap();

    return true;
}

void Consoleui::DrawMap()
{
    if(m_pModel)
    {
        for(unsigned int y = 0; y < m_pModel->GetHeight(); ++y)
        {
            for(unsigned int x = 0; x < m_pModel->GetWidth(); ++x)
                cout << CellToChar(m_pModel->GetCell(x, y));
            cout << endl;
        }
    }
}
