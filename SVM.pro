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

LIBS += -L/usr/local/lib -L/usr/local/cuda/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_xfeatures2d \
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
    mylabel.h \
    libsvm/libsvm.h \
    helper_cuda.h \
    helper_string.h

FORMS    += mainwindow.ui

QMAKE_CXXFLAGS += -std=c++17
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp

CUDA_SOURCES += ./svmCUDA.cu
CUDA_SOURCES += ./cudasvm.cu

CUDA_SDK = "/usr/local/cuda-8.0/"   # Path to cuda SDK install
CUDA_DIR = "/usr/local/cuda-8.0/"            # Path to cuda toolkit install

SYSTEM_NAME = unix         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_21           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math

INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib/

CUDA_OBJECTS_DIR = ./

# Add the necessary libraries
CUDA_LIBS = -lcuda -lcudart

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += $$CUDA_LIBS

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
