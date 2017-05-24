#include <QtGui>
#include <QBrush>
#include <QPen>
#include "renderarea.hpp"
#include "mainwindow.hpp"

RenderArea::RenderArea(QWidget* pParent, Model* pModel)
    : QWidget(pParent)
    , m_pModel(pModel)
    , m_nCellSize(3)
    , m_nStartX(0)
    , m_nStartY(0)
{
    setBackgroundRole(QPalette::Base);
    setAutoFillBackground(true);
}

QSize RenderArea::minimumSizeHint() const
{
    return QSize(600, 600);
}

unsigned int RenderArea::GetCellSize()
{
    return m_nCellSize;
}

void RenderArea::SetCellSize(unsigned int nSize)
{
    m_nCellSize = nSize;
}

void RenderArea::SetScroll(unsigned int x, unsigned int y)
{
    m_nStartX = x;
    m_nStartY = y;
}

QPoint RenderArea::GetScroll() const
{
    return QPoint(m_nStartX, m_nStartY);
}

void RenderArea::paintEvent(QPaintEvent* /* pEvent */)
{
    QPainter aPainter(this);

    for(unsigned int y = m_nStartY; y < m_pModel->GetHeight() && y < m_nStartY + size().height() / m_nCellSize; ++y)
    {
        for(unsigned int x = m_nStartX; x < m_pModel->GetWidth() && x < m_nStartX + size().width() / m_nCellSize; ++x)
        {
            QRect aRect(m_nCellSize * (x - m_nStartX), m_nCellSize * (y - m_nStartY), m_nCellSize, m_nCellSize);
            QBrush aBrush;
            QPen aPen;
            aPen.setColor(QColor(50, 50, 50));
            switch(m_pModel->GetCell(x, y))
            {
            case Head:
                aBrush = QBrush(QColor(0, 0, 255));
                break;
            case Tail:
                aBrush = QBrush(QColor(255, 0, 0));
                break;
            case Conductor:
                aBrush = QBrush(QColor(255, 255, 0));
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

void RenderArea::mouseReleaseEvent(QMouseEvent* pEvent)
{
    unsigned int x = pEvent->x() / m_nCellSize + m_nStartX;
    unsigned int y = pEvent->y() / m_nCellSize + m_nStartY;

    Cell eOld = m_pModel->GetCell(x, y);
    Cell eNew = eOld == Empty ? Conductor : Cell(eOld - 1);
    m_pModel->SetCell(x, y, eNew);
    update();

    QWidget::mouseReleaseEvent(pEvent);
}

void RenderArea::wheelEvent(QWheelEvent* pEvent)
{
    int nDelta = pEvent->delta();
    QPoint aPoint = pEvent->pos();
    int nOldCellSize = m_nCellSize;

    if(nDelta > 0 && m_nCellSize < 30)
        m_nCellSize++;
    else if(nDelta < 0 && m_nCellSize > 3)
        m_nCellSize--;

    // center view in the cursor position
    if(nOldCellSize != m_nCellSize)// && nDelta > 0)
    {
        int nNewX = m_nStartX + ((aPoint.x() / m_nCellSize) - (size().width() / 2 / m_nCellSize)) ;
        int nNewY = m_nStartY + ((aPoint.y() / m_nCellSize) - (size().height() / 2 / m_nCellSize)) ;

        if(nNewX >= 0 && nNewX < (int)m_pModel->GetWidth() - size().width() / m_nCellSize)
            m_nStartX = nNewX;
        if(nNewY >= 0 && nNewY < (int)m_pModel->GetHeight() - size().height() / m_nCellSize)
            m_nStartY = nNewY;
    }

    update();
    MainWindow* pWindow = static_cast<MainWindow*>(parentWidget()->parentWidget());
    pWindow->resizeEvent(new QResizeEvent(pWindow->size(), pWindow->size()));

    QWidget::wheelEvent(pEvent);
}
