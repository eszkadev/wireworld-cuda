#include <simulatorcpp.hpp>
#include <memory.h>

static unsigned int lcl_countHeads(Map pMap, int x, int y, int nWidth, int nHeight)
{
    unsigned int nCount = 0;

    bool bRight = x + 1 < nWidth;
    bool bLeft = x - 1 >= 0;
    bool bTop = y + 1 < nHeight;
    bool bBottom = y - 1 >= 0;

    if(bRight)
    {
        if(pMap[(x + 1) * nHeight + y] == Head)
            nCount++;
        if(bTop && pMap[(x + 1) * nHeight + y + 1] == Head)
            nCount++;
        if(bBottom && pMap[(x + 1) * nHeight + y - 1] == Head)
            nCount++;
    }

    if(bLeft)
    {
        if(pMap[(x - 1) * nHeight + y] == Head)
            nCount++;
        if(bTop && pMap[(x - 1) * nHeight + y + 1] == Head)
            nCount++;
        if(bBottom && pMap[(x - 1) * nHeight + y - 1] == Head)
            nCount++;
    }

    if(bTop && pMap[x * nHeight + y + 1] == Head)
        nCount++;

    if(bBottom && pMap[x * nHeight + y - 1] == Head)
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

void SimulatorCPP::Step(int n)
{
    for(int nStep = 0; nStep < n; ++nStep)
    {
        // first create new map in the m_pModel and copy old state
        Map pNew = m_pModel->GetMap();
        Map pOld = new Cell[m_pModel->GetWidth() * m_pModel->GetHeight()];

        for(unsigned int i = 0; i < m_pModel->GetWidth(); ++i)
            memcpy(pOld + i * m_pModel->GetHeight(), pNew + 1 + (i + 1) * (m_pModel->GetHeight() + 2), m_pModel->GetHeight() * sizeof(Cell));

        // now do simulation
        for(unsigned int x = 0; x < m_pModel->GetWidth(); ++x)
        {
            for(unsigned int y = 0; y < m_pModel->GetHeight(); ++y)
            {
                switch(pOld[x * m_pModel->GetHeight() + y])
                {
                case Head:
                    pNew[1 + (x + 1) * (m_pModel->GetHeight() + 2) + y] = Tail;
                    break;
                case Tail:
                    pNew[1 + (x + 1) * (m_pModel->GetHeight() + 2) + y] = Conductor;
                    break;
                case Conductor:
                    {
                        unsigned int nHeads = lcl_countHeads(pOld, x, y, m_pModel->GetWidth(), m_pModel->GetHeight());
                        if(nHeads == 1 || nHeads == 2)
                            pNew[1 + (x + 1) * (m_pModel->GetHeight() + 2) + y] = Head;
                        else
                            pNew[1 + (x + 1) * (m_pModel->GetHeight() + 2) + y] = Conductor;
                    }
                    break;
                default:
                    pNew[1 + (x + 1) * (m_pModel->GetHeight() + 2) + y] = Empty;
                    break;
                }
            }
        }

        // delete old
        delete[] pOld;
    }
}
