#include <model.hpp>

Model::Model(unsigned int nWidth, unsigned int nHeight)
{
    if(m_pMap)
    {
        for(unsigned int i = 0; i < m_nWidth; i++)
            delete[] m_pMap[i];
        delete[] m_pMap;
    }

    m_pMap = new Cell*[nWidth];
    for(unsigned int i = 0; i < nWidth; i++)
        m_pMap[i] = new Cell[nHeight];

    m_nHeight = nHeight;
    m_nWidth = nWidth;
}

Map Model::GetMap()
{
    return m_pMap;
}

Cell Model::GetCell(unsigned int nWidth, unsigned int nHeight)
{
    return m_pMap[nWidth][nHeight];
}

void Model::SetCell(unsigned int nWidth, unsigned int nHeight, Cell eCell)
{
    m_pMap[nWidth][nHeight] = eCell;
}
