#include "mainwindow.hpp"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QPainter>
#include <renderarea.hpp>
#include <QTimer>
#include <QPushButton>
#include <simulatorcpp.hpp>
#include <simulatorcuda.hpp>
#include <sstream>
#include <ctime>

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

    m_pUi->openButton->connect(m_pUi->openButton, SIGNAL(clicked()), this, SLOT(Open()));
    m_pUi->stepButton->connect(m_pUi->stepButton, SIGNAL(clicked()), this, SLOT(Step()));
    m_pUi->stepsButton->connect(m_pUi->stepsButton, SIGNAL(clicked()), this, SLOT(Steps()));
    m_pUi->horizontalScrollBar->connect(m_pUi->horizontalScrollBar, SIGNAL(sliderMoved(int)), this, SLOT(UpdateScroll()));
    m_pUi->verticalScrollBar->connect(m_pUi->verticalScrollBar, SIGNAL(sliderMoved(int)), this, SLOT(UpdateScroll()));
    m_pUi->cellSlider->connect(m_pUi->cellSlider, SIGNAL(sliderMoved(int)), this, SLOT(UpdateCellSize()));
    m_pUi->cellSlider->setMinimum(3);
    m_pUi->cellSlider->setMaximum(30);
    m_pUi->implementationComboBox->connect(m_pUi->implementationComboBox, SIGNAL(currentIndexChanged(QString)), this, SLOT(ChangeImplementation()));

    m_pStatusLabel = new QLabel(m_pUi->statusBar);
    m_pUi->statusBar->addWidget(m_pStatusLabel);
}

MainWindow::~MainWindow()
{
    delete m_pUi;
}

void MainWindow::Open()
{
    if(!m_pModel)
        m_pModel = new Model();
    std::string sFileName = QFileDialog::getOpenFileName(this, "Open File", "", "Files (*.txt)").toStdString();
    m_pModel->LoadModel(sFileName);
    m_pRenderArea->update();
}

void MainWindow::Step()
{
    m_pSimulator->Step();
    m_pRenderArea->update();
}

void MainWindow::Steps()
{
    unsigned int nCount = m_pUi->runSpinBox->value();
    clock_t nBegin = clock();

    for(unsigned int i = 0; i < nCount; ++i)
    {
        m_pSimulator->Step();
        m_pRenderArea->update();

        std::stringstream ss;
        ss << i << "/" << nCount;

        m_pStatusLabel->setText(ss.str().c_str());
    }

    clock_t nEnd = clock();
    double nElapsed = double(nEnd - nBegin) / CLOCKS_PER_SEC;

    std::stringstream ss;
    ss << "Done. Execution time: " << nElapsed << " s";
    m_pStatusLabel->setText(ss.str().c_str());
}

void MainWindow::resizeEvent(QResizeEvent*)
{
    UpdateScollbars();
    m_pUi->cellSlider->setSliderPosition(m_pRenderArea->GetCellSize());
}

void MainWindow::UpdateScollbars()
{
    QPoint aPos = m_pRenderArea->GetScroll();

    m_pUi->horizontalScrollBar->setMinimum(0);
    int nMaxWidth = m_pModel->GetWidth() - m_pRenderArea->width()/m_pRenderArea->GetCellSize();
    m_pUi->horizontalScrollBar->setMaximum(nMaxWidth >= 0 ? nMaxWidth : 0);
    m_pUi->horizontalScrollBar->setValue(aPos.x());

    m_pUi->verticalScrollBar->setMinimum(0);
    int nMaxHeight = m_pModel->GetHeight() - m_pRenderArea->height()/m_pRenderArea->GetCellSize();
    m_pUi->verticalScrollBar->setMaximum(nMaxHeight >= 0 ? nMaxHeight : 0);
    m_pUi->verticalScrollBar->setValue(aPos.y());
}

void MainWindow::UpdateCellSize()
{
    m_pRenderArea->SetCellSize(m_pUi->cellSlider->sliderPosition());
    UpdateScollbars();
    m_pRenderArea->update();
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

void MainWindow::ChangeImplementation()
{
    std::string sImpl = m_pUi->implementationComboBox->currentText().toStdString();
    if(sImpl == "CPU")
    {
        delete m_pSimulator;
        m_pSimulator = new SimulatorCPP(m_pModel);
    }
    else if(sImpl == "CUDA")
    {
        delete m_pSimulator;
        m_pSimulator = new SimulatorCUDA(m_pModel);
    }

    m_pSimulator->Setup();
}
