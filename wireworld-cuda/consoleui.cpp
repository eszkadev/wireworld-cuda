#include <consoleui.hpp>
#include <model.hpp>
#include <simulatorcpp.hpp>
#include <iostream>

using namespace std;

void lcl_sleep()
{
    int j = 1;
    int k = 0;
    while(1)
    {
        j++;
        if(j == 25000000)
        {
            k++;
            j = 0;
        }
        if(k == 10)
            break;
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

    m_pModel = new Model();
    m_pModel->LoadModel("model2.txt");

    m_pSimulator = new SimulatorCPP(m_pModel);
    m_pSimulator->Setup();

    // TODO: event loop
    for(int i = 0; i < 30; i++)
    {
        cout << "-----------------------------\n";
        DrawMap();
        m_pSimulator->Step(1);
        lcl_sleep();
    }

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
