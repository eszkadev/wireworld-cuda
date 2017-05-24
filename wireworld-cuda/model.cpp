#include <model.hpp>
#include <fstream>

char CellToChar(Cell eCell)
{
    switch(eCell)
    {
    case Empty:
        return '.';
    case Head:
        return 'H';
    case Tail:
        return 'T';
    case Conductor:
        return '+';
    default:
        throw;
    }
}

Cell CharToCell(char cChar)
{
    switch(cChar)
    {
    case '.':
        return Empty;
    case 'H':
        return Head;
    case 'T':
        return Tail;
    case '+':
        return Conductor;
    default:
        throw;
    }
}

Model::Model()
    : m_pMap(NULL)
    , m_nWidth(0)
    , m_nHeight(0)
{
}

Model::Model(unsigned int nWidth, unsigned int nHeight)
{
    NewModel(nWidth, nHeight);
}

void Model::NewModel(unsigned int nWidth, unsigned int nHeight)
{
    delete[] m_pMap;

    m_pMap = new Cell[(nWidth + 2)*(nHeight + 2)];

    m_nHeight = nHeight;
    m_nWidth = nWidth;
}

void Model::LoadModel(const std::string& rFilePath)
{
    std::ifstream aStream(rFilePath.c_str(), std::ios::binary);
    if(aStream.is_open())
    {
        char cChar;
        aStream >> m_nWidth;
        aStream >> m_nHeight;

        NewModel(m_nWidth, m_nHeight);

        for(unsigned int y = 0; y < m_nHeight; ++y)
        {
            for(unsigned int x = 0; x < m_nWidth; ++x)
            {
                aStream >> cChar;
                Cell eCell = CharToCell(cChar);
                SetCell(x, y, eCell);
            }
        }

        aStream.close();
    }
}

Map Model::GetMap()
{
    return m_pMap;
}

unsigned int Model::GetWidth()
{
    return m_nWidth;
}

unsigned int Model::GetHeight()
{
    return m_nHeight;
}

Cell Model::GetCell(unsigned int nWidth, unsigned int nHeight)
{
    return m_pMap[(nWidth + 1) * (m_nHeight + 2) + nHeight + 1];
}

void Model::SetCell(unsigned int nWidth, unsigned int nHeight, Cell eCell)
{
    m_pMap[(nWidth + 1) * (m_nHeight + 2) + nHeight + 1] = eCell;
}
