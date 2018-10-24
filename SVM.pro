#-------------------------------------------------
#
# Project created by QtCreator 2018-10-09T17:44:40
#
#-------------------------------------------------

QT       += core gui sql network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = SVM
TEMPLATE = app

INCLUDEPATH += /usr/local/include/opencv

INCLUDEPATH += /usr/lib/x86_64-linux-gnu

LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_xfeatures2d \
        -lopencv_flann -lopencv_stitching -lopencv_objdetect -lopencv_video -lopencv_ml -fopenmp -lavutil -lavcodec -lavformat -lswscale

LIBS += -lopencv_cudawarping -lopencv_cudacodec \
        -lopencv_cudabgsegm -lopencv_cudastereo -lopencv_cudalegacy \
        -lopencv_cudaobjdetect -lopencv_cudaarithm -lopencv_cudaimgproc -fopenmp

SOURCES += main.cpp\
        mainwindow.cpp \
    libsvm/svm.cpp \
    libsvm/svm-scale.c \
    libsvm/svm-train.c \
    libsvm/svm-predict.c \
    mylabel.cpp

HEADERS  += mainwindow.h \
    libsvm/svm.h \
    mylabel.h

FORMS    += mainwindow.ui

QMAKE_CXXFLAGS += -std=c++17
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
