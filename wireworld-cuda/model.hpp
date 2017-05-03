#ifndef MODEL_HPP
#define MODEL_HPP

#include <string>

typedef enum Cell
{
    Empty,
    Head,
    Tail,
    Conductor
} Cell;

char CellToChar(Cell eCell);
Cell CharToCell(char cChar);

typedef Cell** Map;

class Model
{
    Map m_pMap;
    unsigned int m_nWidth;
    unsigned int m_nHeight;

public:
    Model();
    Model(unsigned int nWidth, unsigned int nHeight);
    void NewModel(unsigned int nWidth, unsigned int nHeight);
    void LoadModel(const std::string& rFilePath);

    Map GetMap();
    unsigned int GetWidth();
    unsigned int GetHeight();
    Cell GetCell(unsigned int nWidth, unsigned int nHeight);
    void SetCell(unsigned int nWidth, unsigned int nHeight, Cell eCell);
};

#endif // MODEL_HPP
