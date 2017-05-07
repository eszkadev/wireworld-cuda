#ifndef RENDERAREA_HPP
#define RENDERAREA_HPP

#include <QWidget>
#include <model.hpp>

class RenderArea : public QWidget
{
    Q_OBJECT

public:
    RenderArea(QWidget* pParent = 0, Model* pModel = 0);

    QSize minimumSizeHint() const;
    void SetScroll(unsigned int x, unsigned int y);
    unsigned int GetCellSize();
    void SetCellSize(unsigned int nSize);

protected:
    virtual void mouseReleaseEvent(QMouseEvent* pEvent);
    virtual void wheelEvent(QWheelEvent* pEvent);

protected:
    void paintEvent(QPaintEvent* pEvent);
    Model* m_pModel;
    int m_nCellSize;
    unsigned int m_nStartX;
    unsigned int m_nStartY;
};

#endif // RENDERAREA_HPP
