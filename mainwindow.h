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
#include <QMutableStringListIterator>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <unistd.h>
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
#include <opencv2/core/cuda.hpp>//!< Probando
#include <opencv2/objdetect/objdetect.hpp> //!< Probando
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaobjdetect.hpp> //!< Probando
#include <opencv2/videoio.hpp>
#include <opencv2/ml.hpp>
#include <omp.h>

#include <libsvm/svm.h>
#include <libsvm/libsvm.h>

#include "mylabel.h"

using namespace std;
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
    std::string currentInputPath;
    std::string currentOutputPath;
    std::mutex mtx;
    QDir directoryIn;
    QDir directoryTrainSet1;
    QDir directoryTrainSet2;
    QDir directoryTestSet1;
    QDir directoryTestSet2;
    QStringList qslValidExtensions;
    ///Variabls necessaries for libsvm predict
    struct svm_model* model;
    struct svm_node *x;
    int predict_probability=0;
    int max_nr_attr = 64;
    char *line = NULL;
    static int max_line_len;
    constexpr static int (*info)(const char *fmt,...) = &printf;
    Size trainingPadding = Size(0, 0);
    Size winStride = Size(8, 8);

    bool getFlagPython() const;
    void setFlagPython(bool value);

    bool getFlagUbicationDefined() const;
    void setFlagUbicationDefined(bool value);

    int getCurrentBeginingFrameTV() const;
    void setCurrentBeginingFrameTV(int value);

    int getCurrentBeginingFramePanoramic() const;
    void setCurrentBeginingFramePanoramic(int value);

    int getCurrentSingleFramePanoramic() const;
    void setCurrentSingleFramePanoramic(int value);

    int getCurrentSingleChunkPanoramic() const;
    void setCurrentSingleChunkPanoramic(int value);

    int getCurrentSingleFrameTV() const;
    void setCurrentSingleFrameTV(int value);

    int getCurrentSingleChunkTV() const;
    void setCurrentSingleChunkTV(int value);

    std::string getStrFileNamePanoramic() const;
    void setStrFileNamePanoramic(const std::string &value);

    std::string getStrFileNameTV() const;
    void setStrFileNameTV(const std::string &value);

    std::string getStrFileNameGameConfig() const;
    void setStrFileNameGameConfig(const std::string &value);

    std::string getStrGameConfigPath() const;
    void setStrGameConfigPath(const std::string &value);

    std::string getStrGameId() const;
    void setStrGameId(const std::string &value);

    std::string getCurrentOutputPath() const;
    void setCurrentOutputPath(const std::string &value);

    std::string getCurrentInputPath() const;
    void setCurrentInputPath(const std::string &value);

    void updateVideoLabelTV();
    void updateVideoLabelPanoramic();
    void updateGUI();
    void setValidExtensions();
    char *readline(FILE *input);
    int svm_predict_main(int argc, char **argv);
    QDir getDirectoryIn() const;
    void setDirectoryIn(const QDir &value);

    int getFilesInDirectory(QStringList &Files);
    QDir getDirectoryTrainSet1() const;
    void setDirectoryTrainSet1(const QDir &value);

    QDir getDirectoryTrainSet2() const;
    void setDirectoryTrainSet2(const QDir &value);

    QDir getDirectoryTestSet1() const;
    void setDirectoryTestSet1(const QDir &value);

    QDir getDirectoryTestSet2() const;
    void setDirectoryTestSet2(const QDir &value);

    int getFilesInDirectory(QDir directory, QStringList &Files);
    void calculateFeaturesFromInput(const string  imageFilename, vector<float> &featureVector, HOGDescriptor &hog);
    void detectTrainingSetTest(const HOGDescriptor &hog, const double hitThreshold, QStringList &posFileNames, QStringList &negFileNames);
    void detectTest(const HOGDescriptor &hog, const double hitThreshold, Mat &imageData);
private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

    void on_pushButton_5_clicked();

    void on_pushButton_6_clicked();

    void on_pushButton_7_clicked();

    void on_pushButton_8_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
