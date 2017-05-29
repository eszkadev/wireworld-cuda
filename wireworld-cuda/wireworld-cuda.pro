#-------------------------------------------------
#
# Project created by QtCreator 2017-05-03T21:02:18
#
#-------------------------------------------------

PROJECT_DIR = $$system(pwd)
OBJECTS_DIR = $$PROJECT_DIR/Obj

CUDA_SOURCES += cudakernel.cu

CUDA_DIR = /usr/local/cuda
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += $$PROJECT_DIR
QMAKE_LIBDIR += $$CUDA_DIR/lib64

LIBS += -lcudart
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

cuda.commands = $$CUDA_DIR/bin/nvcc -ccbin g++ -m64 -g -G -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -ccbin g++ -m64 -g -G -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

QMAKE_EXTRA_COMPILERS += cuda

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = wireworld-cuda
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    model.cpp \
    consoleui.cpp \
    simulatorcpp.cpp \
    renderarea.cpp \
    simulatorcuda.cpp

HEADERS  += mainwindow.hpp \
    model.hpp \
    consoleui.hpp \
    simulator.hpp \
    simulatorcpp.hpp \
    renderarea.hpp \
    simulatorcuda.hpp \
    devicesinfo.h

FORMS    += mainwindow.ui
