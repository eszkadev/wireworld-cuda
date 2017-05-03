#define USE_QT

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

int main(int argc, char* argv[])
{
    return 0;
}

#endif
