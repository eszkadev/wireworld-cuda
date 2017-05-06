#include <QtGui>
#include <QBrush>
#include <QPen>
#include "renderarea.hpp"

RenderArea::RenderArea(QWidget* pParent, Model* pModel)
    : QWidget(pParent)
    , m_pModel(pModel)
    , m_nCellSize(3)
{
    setBackgroundRole(QPalette::Base);
    setAutoFillBackground(true);
}

QSize RenderArea::minimumSizeHint() const
{
    return QSize(100, 100);
}

QSize RenderArea::sizeHint() const
{
    return QSize(400, 200);
}

void RenderArea::paintEvent(QPaintEvent * /* pEvent */)
{
    QPainter aPainter(this);

    for(unsigned int y = 0; y < m_pModel->GetHeight(); ++y)
    {
        for(unsigned int x = 0; x < m_pModel->GetWidth(); ++x)
        {
            QRect aRect(m_nCellSize * x, m_nCellSize * y, m_nCellSize, m_nCellSize);
            QBrush aBrush;
            QPen aPen;
            aPen.setColor(QColor(255,255,255));
            switch(m_pModel->GetCell(x, y))
            {
            case Head:
                aBrush = QBrush(QColor(255, 0, 0));
                break;
            case Tail:
                aBrush = QBrush(QColor(0, 0, 255));
                break;
            case Conductor:
                aBrush = QBrush(QColor(0, 255, 255));
                break;
            default:
                aBrush = QBrush(QColor(0, 0, 0));
                break;
            }

            aPainter.setPen(aPen);
            aPainter.setBrush(aBrush);
            aPainter.drawRect(aRect);
        }
    }
}
