#include "mainwindow.hpp"
#include "ui_mainwindow.h"
#include <QPainter>
#include <renderarea.hpp>
#include <QTimer>
#include <QPushButton>
#include <simulatorcpp.hpp>

MainWindow::MainWindow(QWidget* pParent)
    : QMainWindow(pParent)
    , m_pUi(new Ui::MainWindow)
    , m_pModel(NULL)
    , m_pSimulator(NULL)
{
    m_pUi->setupUi(this);

    m_pModel = new Model();
    m_pModel->LoadModel("model2.txt");

    m_pSimulator = new SimulatorCPP(m_pModel);
    m_pSimulator->Setup();

    m_pRenderArea = new RenderArea(this, m_pModel);
    m_pUi->gridLayout->addWidget(m_pRenderArea);

    m_pUi->stepButton->connect(m_pUi->stepButton, SIGNAL(clicked()), this, SLOT(Step()));
    m_pUi->horizontalScrollBar->connect(m_pUi->horizontalScrollBar, SIGNAL(sliderMoved(int)), this, SLOT(UpdateScroll()));
    m_pUi->verticalScrollBar->connect(m_pUi->verticalScrollBar, SIGNAL(sliderMoved(int)), this, SLOT(UpdateScroll()));
}

MainWindow::~MainWindow()
{
    delete m_pUi;
}

void MainWindow::Step()
{
    m_pSimulator->Step();
    m_pRenderArea->update();
}

void MainWindow::resizeEvent(QResizeEvent*)
{
    m_pUi->horizontalScrollBar->setMinimum(0);
    m_pUi->horizontalScrollBar->setMaximum(m_pModel->GetWidth() - m_pRenderArea->width()/m_pRenderArea->GetCellSize());
    m_pUi->verticalScrollBar->setMinimum(0);
    m_pUi->verticalScrollBar->setMaximum(m_pModel->GetHeight() - m_pRenderArea->height()/m_pRenderArea->GetCellSize());
}

void MainWindow::UpdateScroll()
{
    unsigned int x;
    unsigned int y;

    x = m_pUi->horizontalScrollBar->sliderPosition();
    y = m_pUi->verticalScrollBar->sliderPosition();

    m_pRenderArea->SetScroll(x, y);
    m_pRenderArea->update();
}
