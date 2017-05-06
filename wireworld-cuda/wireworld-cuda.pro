#-------------------------------------------------
#
# Project created by QtCreator 2017-05-03T21:02:18
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = wireworld-cuda
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    model.cpp \
    consoleui.cpp \
    simulatorcpp.cpp \
    renderarea.cpp

HEADERS  += mainwindow.hpp \
    model.hpp \
    consoleui.hpp \
    simulator.hpp \
    simulatorcpp.hpp \
    renderarea.hpp

FORMS    += mainwindow.ui
