#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <QMainWindow>
#include <renderarea.hpp>
#include <model.hpp>
#include <simulator.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public slots:
    void Step();

public:
    explicit MainWindow(QWidget* pParent = 0);
    ~MainWindow();

private:
    Ui::MainWindow* m_pUi;
    RenderArea* m_pRenderArea;
    Model* m_pModel;
    Simulator* m_pSimulator;
};

#endif // MAINWINDOW_HPP
