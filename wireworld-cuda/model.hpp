#ifndef MODEL_HPP
#define MODEL_HPP

enum Cell
{
    Empty,
    Head,
    Tail,
    Conductor
};

typedef Cell** Map;

class Model
{
    Map m_pMap;
    unsigned int m_nWidth;
    unsigned int m_nHeight;

public:
    Model(unsigned int nWidth, unsigned int nHeight);

    Map GetMap();
    unsigned int GetWidth();
    unsigned int GetHeight();
    Cell GetCell(unsigned int nWidth, unsigned int nHeight);
    void SetCell(unsigned int nWidth, unsigned int nHeight, Cell eCell);
};

#endif // MODEL_HPP
