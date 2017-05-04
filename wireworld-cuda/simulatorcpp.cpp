#include <simulatorcpp.hpp>

static unsigned int lcl_countHeads(Map pMap, int x, int y, int nWidth, int nHeight)
{
    unsigned int nCount = 0;

    bool bRight = x + 1 < nWidth;
    bool bLeft = x - 1 >= 0;
    bool bTop = y + 1 < nHeight;
    bool bBottom = y - 1 >= 0;

    if(bRight)
    {
        if(pMap[x + 1][y] == Head)
            nCount++;
        if(bTop && pMap[x + 1][y + 1] == Head)
            nCount++;
        if(bBottom && pMap[x + 1][y - 1] == Head)
            nCount++;
    }

    if(bLeft)
    {
        if(pMap[x - 1][y] == Head)
            nCount++;
        if(bTop && pMap[x - 1][y + 1] == Head)
            nCount++;
        if(bBottom && pMap[x - 1][y - 1] == Head)
            nCount++;
    }

    if(bTop && pMap[x][y + 1] == Head)
        nCount++;

    if(bBottom && pMap[x][y - 1] == Head)
        nCount++;

    return nCount;
}

SimulatorCPP::SimulatorCPP(Model* pModel)
    : Simulator(pModel)
{

}

void SimulatorCPP::Setup()
{

}

void SimulatorCPP::Step()
{
    // first create new map in the m_pModel and copy old state
    Map pNew = m_pModel->GetMap();
    Map pOld = new Cell*[m_pModel->GetWidth()];

    for(unsigned int i = 0; i < m_pModel->GetWidth(); ++i)
    {
        pOld[i] = pNew[i];
        pNew[i] = new Cell[m_pModel->GetHeight()];
    }

    // now do simulation
    for(unsigned int x = 0; x < m_pModel->GetWidth(); ++x)
    {
        for(unsigned int y = 0; y < m_pModel->GetHeight(); ++y)
        {
            switch(pOld[x][y])
            {
            case Head:
                pNew[x][y] = Tail;
                break;
            case Tail:
                pNew[x][y] = Conductor;
                break;
            case Conductor:
                {
                    unsigned int nHeads = lcl_countHeads(pOld, x, y, m_pModel->GetWidth(), m_pModel->GetHeight());
                    if(nHeads == 1 || nHeads == 2)
                        pNew[x][y] = Head;
                    else
                        pNew[x][y] = Conductor;
                }
                break;
            default:
                pNew[x][y] = Empty;
                break;
            }
        }
    }

    // delete old
    for(unsigned int i = 0; i < m_pModel->GetWidth(); ++i)
    {
        delete[] pOld[i];
    }
    delete[] pOld;
}
