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

friend class RenderArea;

public slots:
    void Step();
    void Steps();
    void UpdateScroll();
    void UpdateCellSize();
    void ChangeImplementation();

public:
    explicit MainWindow(QWidget* pParent = 0);
    ~MainWindow();

protected:
    virtual void resizeEvent(QResizeEvent*);

private:
    void UpdateScollbars();

private:
    Ui::MainWindow* m_pUi;
    RenderArea* m_pRenderArea;
    Model* m_pModel;
    Simulator* m_pSimulator;
};

#endif // MAINWINDOW_HPP
