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
    QSize sizeHint() const;

protected:
    void paintEvent(QPaintEvent* pEvent);
    Model* m_pModel;
    int m_nCellSize;
};

#endif // RENDERAREA_HPP
