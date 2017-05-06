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
