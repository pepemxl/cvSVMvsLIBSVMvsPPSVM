#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QButtonGroup>
#include <QDebug>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QMessageBox>
#include <QRect>
#include <QString>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QWidget>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlRecord>
#include <QSqlField>
#include <QSqlError>
#include <QEventLoop>
#include <QThread>
#include <QMutex>
#include <QHostAddress>
#include <QHostInfo>
#include <QNetworkInterface>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <ratio>
#include <iomanip>
#include <fstream>
#include <stack>
#include <string>
#include <thread>
#include <vector>
#include <exception>
#include <sys/time.h>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect/objdetect.hpp> //<!Probando
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/ml.hpp>
#include <omp.h>

#include <libsvm/svm.h>

#include "mylabel.h"

using namespace cv;
using namespace cv::ml;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void runPythonScript(char *path, char *fileName, char *extension);
    bool flagPython;
    bool flagUbicationDefined;
    FILE *fout;
    QMessageBox QMB_aviso;
    QMessageBox QMB_pregunta;
    MyLabel *ptrTVVideoLabel;
    MyLabel *ptrPanoramicVideoLabel;
    cv::VideoCapture videoCapturatorPanoramic;
    cv::VideoCapture videoCapturatorTV;
    cv::Mat tvFrame;
    cv::Mat panoramicFrame;
    int currentBeginingFrameTV;
    int currentBeginingFramePanoramic;
    int currentSingleFramePanoramic;
    int currentSingleChunkPanoramic;
    int currentSingleFrameTV;
    int currentSingleChunkTV;
    std::string strFileNamePanoramic;
    std::string strFileNameTV;
    std::string strFileNameGameConfig;
    std::string strGameConfigPath;
    std::string strGameId;
    std::string currentOutputPath;
    bool getFlagPython() const;
    void setFlagPython(bool value);

private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
