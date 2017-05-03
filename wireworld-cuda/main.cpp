//#define USE_QT

#ifdef USE_QT

// GUI

#include "mainwindow.hpp"
#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}

#else

// CONSOLE

#include <consoleui.hpp>

int main(int argc, char* argv[])
{
    Consoleui cui;

    if(cui.Run())
        return 0;
    else
        return 1;
}

#endif
