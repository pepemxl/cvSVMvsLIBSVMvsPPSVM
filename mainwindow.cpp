#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setFlagPython(false);
    this->setMinimumWidth(1520);
    this->setMinimumHeight(480+240);
    this->setMaximumHeight(480+240);
    this->setStyleSheet(QString::fromStdString("background-color: #505050;"));
    this->ptrPanoramicVideoLabel = new MyLabel(this);
    this->ptrPanoramicVideoLabel->setText(QString::fromStdString("Panorámica"));
    this->ptrPanoramicVideoLabel->setId(1);
    this->ptrPanoramicVideoLabel->setAlignment(Qt::AlignLeft);
    this->ptrPanoramicVideoLabel->setAlignment(Qt::AlignTop);
    this->ptrPanoramicVideoLabel->setStyleSheet(QString::fromStdString("background-color: #282828; border: 2px solid #464646;"));
    this->ptrPanoramicVideoLabel->setMinimumWidth(1280);
    this->ptrPanoramicVideoLabel->setMinimumHeight(240);
    this->ptrPanoramicVideoLabel->setMaximumWidth(1280);
    this->ptrPanoramicVideoLabel->setMaximumHeight(240);
    this->ptrTVVideoLabel = new MyLabel(this);
    this->ptrTVVideoLabel->setText(QString::fromStdString("TV"));
    this->ptrTVVideoLabel->setId(1);
    this->ptrTVVideoLabel->setAlignment(Qt::AlignLeft);
    this->ptrTVVideoLabel->setAlignment(Qt::AlignTop);
    this->ptrTVVideoLabel->setStyleSheet(QString::fromStdString("background-color: #282828; border: 2px solid #464646;"));
    this->ptrTVVideoLabel->setMinimumWidth(640);
    this->ptrTVVideoLabel->setMinimumHeight(480);
    this->ptrTVVideoLabel->setMaximumWidth(640);
    this->ptrTVVideoLabel->setMaximumHeight(480);
    ui->horizontalLayoutPanoramica->addWidget(this->ptrPanoramicVideoLabel);
    ui->horizontalLayoutTV->addWidget(this->ptrTVVideoLabel);
    ui->horizontalLayoutTV->addStretch();
    this->setCurrentSingleFramePanoramic(0);
    this->setCurrentSingleChunkPanoramic(0);
    this->setValidExtensions();
}

MainWindow::~MainWindow()
{
    delete ui;
}

/**
 * @brief      Sets the valid extensions.
 */
void MainWindow::setValidExtensions(){
    this->qslValidExtensions.push_back("*.jpg");
    this->qslValidExtensions.push_back("*.JPG");
    this->qslValidExtensions.push_back("*.png");
    this->qslValidExtensions.push_back("*.PNG");
}
/**
 * @brief      Run python script
 *
 * @param      path       The path
 * @param      fileName   The file name
 * @param      extension  The extension
 */
void MainWindow::runPythonScript(char *path,char *fileName,char *extension){
    if(!this->getFlagPython()){
        fout = popen("/home/pepe/anaconda3/envs/opencv/bin/python", "w");
        fprintf(fout,"import numpy\n");
        fprintf(fout,"import numpy as np\n");
        fprintf(fout,"from sklearn.svm import SVC\n");
        fprintf(fout,"import cv2\n");
        fprintf(fout,"from scipy import signal\n");
        fprintf(fout,"from sklearn.cross_validation import train_test_split\n");
        fprintf(fout,"from imutils.object_detection import non_max_suppression\n");
        fprintf(fout,"import math\n");
        fprintf(fout,"import scipy.misc\n");
        fprintf(fout,"from skimage.feature import hog\n");
        fprintf(fout,"import os\n");
        fprintf(fout,"import glob\n");
        fprintf(fout,"from sklearn.externals import joblib\n");
        fprintf(fout,"import pickle\n");
        fprintf(fout,"\n");
        fprintf(fout,"MODEL_FILENAME = \"./model\"\n");
        fprintf(fout,"\n");
        fprintf(fout,"hogcv2 = cv2.HOGDescriptor()\n");
        fprintf(fout,"\n");
        fprintf(fout,"clf = joblib.load('./'+'modelo_svm_scikit.pkl', 'rb')\n");
        fprintf(fout,"\n");
        fprintf(fout,"def hoggify_image(image_file_name,extension,is_color):\n");
        fprintf(fout,"    data=[]\n");
        fprintf(fout,"    file=image_file_name+'.'+extension\n");
        fprintf(fout,"    image = cv2.imread(file, is_color)\n");
        fprintf(fout,"    dim = 128\n");
        fprintf(fout,"    img = cv2.resize(image, (dim,dim), interpolation = cv2.INTER_AREA)\n");
        fprintf(fout,"    img = hogcv2.compute(img)\n");
        fprintf(fout,"    img = np.squeeze(img)\n");
        fprintf(fout,"    data.append(img)\n");
        fprintf(fout,"    return data\n");
        fprintf(fout,"\n");
        this->setFlagPython(true);
    }else{
        fflush(fout);
        fclose(fout);
        fout = popen("/home/pepe/anaconda3/envs/opencv/bin/python", "a+");
    }
    //fprintf(fout,"if __name__=='__main__':\n");
    fprintf(fout,"dataHogSample=hoggify_image(\'%s/%s\',\'%s\',False)\n",path,fileName,extension);
    fprintf(fout,"predicted=clf.predict_proba(dataHogSample)\n");
    fprintf(fout,"print(str(predicted[0,0]))\n");
    fprintf(fout,"with open('./probability.dat', 'w') as f:\n");
    fprintf(fout,"    f.write(str(predicted[0,0]))\n");
    fprintf(fout,"    f.close()\n");
}

/**
 * @brief      Gets the directory test set 2.
 *
 * @return     The directory test set 2.
 */
QDir MainWindow::getDirectoryTestSet2() const
{
    return directoryTestSet2;
}
/**
 * @brief      Sets the directory test set 2.
 *
 * @param[in]  value  The value
 */
void MainWindow::setDirectoryTestSet2(const QDir &value)
{
    directoryTestSet2 = value;
}
/**
 * @brief      Gets the directory test set 1.
 *
 * @return     The directory test set 1.
 */
QDir MainWindow::getDirectoryTestSet1() const
{
    return directoryTestSet1;
}
/**
 * @brief      Sets the directory test set 1.
 *
 * @param[in]  value  The value
 */
void MainWindow::setDirectoryTestSet1(const QDir &value)
{
    directoryTestSet1 = value;
}
/**
 * @brief      Gets the directory train set 2.
 *
 * @return     The directory train set 2.
 */
QDir MainWindow::getDirectoryTrainSet2() const
{
    return directoryTrainSet2;
}
/**
 * @brief      Sets the directory train set 2.
 *
 * @param[in]  value  The value
 */
void MainWindow::setDirectoryTrainSet2(const QDir &value)
{
    directoryTrainSet2 = value;
}
/**
 * @brief      Gets the directory train set 1.
 *
 * @return     The directory train set 1.
 */
QDir MainWindow::getDirectoryTrainSet1() const
{
    return directoryTrainSet1;
}
/**
 * @brief      Sets the directory train set 1.
 *
 * @param[in]  value  The value
 */
void MainWindow::setDirectoryTrainSet1(const QDir &value)
{
    directoryTrainSet1 = value;
}
/**
 * @brief      Gets the directory in.
 *
 * @return     The directory in.
 */
QDir MainWindow::getDirectoryIn() const
{
    return directoryIn;
}
/**
 * @brief      Sets the directory in.
 *
 * @param[in]  value  The value
 */
void MainWindow::setDirectoryIn(const QDir &value)
{
    directoryIn = value;
}
/**
 * @brief      Gets the current input path.
 *
 * @return     The current input path.
 */
std::string MainWindow::getCurrentInputPath() const
{
    return currentInputPath;
}
/**
 * @brief      Sets the current input path.
 *
 * @param[in]  value  The value
 */
void MainWindow::setCurrentInputPath(const std::string &value)
{
    currentInputPath = value;
}
/**
 * @brief      Gets the current output path.
 *
 * @return     The current output path.
 */
std::string MainWindow::getCurrentOutputPath() const
{
    return currentOutputPath;
}
/**
 * @brief      Sets the current output path.
 *
 * @param[in]  value  The value
 */
void MainWindow::setCurrentOutputPath(const std::string &value)
{
    currentOutputPath = value;
}

std::string MainWindow::getStrGameId() const
{
    return strGameId;
}
/**
 * @brief      Sets the string game identifier.
 *
 * @param[in]  value  The value
 */
void MainWindow::setStrGameId(const std::string &value)
{
    strGameId = value;
}
/**
 * @brief      Gets the string game configuration path.
 *
 * @return     The string game configuration path.
 */
std::string MainWindow::getStrGameConfigPath() const
{
    return strGameConfigPath;
}
/**
 * @brief      Sets the string game configuration path.
 *
 * @param[in]  value  The value
 */
void MainWindow::setStrGameConfigPath(const std::string &value)
{
    strGameConfigPath = value;
}
/**
 * @brief      Gets the string file name game configuration.
 *
 * @return     The string file name game configuration.
 */
std::string MainWindow::getStrFileNameGameConfig() const
{
    return strFileNameGameConfig;
}
/**
 * @brief      Sets the string file name game configuration.
 *
 * @param[in]  value  The value
 */
void MainWindow::setStrFileNameGameConfig(const std::string &value)
{
    strFileNameGameConfig = value;
}
/**
 * @brief      Gets the string file name tv.
 *
 * @return     The string file name tv.
 */
std::string MainWindow::getStrFileNameTV() const
{
    return strFileNameTV;
}
/**
 * @brief      Sets the string file name tv.
 *
 * @param[in]  value  The value
 */
void MainWindow::setStrFileNameTV(const std::string &value)
{
    strFileNameTV = value;
}
/**
 * @brief      Gets the string file name panoramic.
 *
 * @return     The string file name panoramic.
 */
std::string MainWindow::getStrFileNamePanoramic() const
{
    return strFileNamePanoramic;
}
/**
 * @brief      Sets the string file name panoramic.
 *
 * @param[in]  value  The value
 */
void MainWindow::setStrFileNamePanoramic(const std::string &value)
{
    strFileNamePanoramic = value;
}

int MainWindow::getCurrentSingleChunkTV() const
{
    return currentSingleChunkTV;
}
/**
 * @brief      Sets the current single chunk tv.
 *
 * @param[in]  value  The value
 */
void MainWindow::setCurrentSingleChunkTV(int value)
{
    currentSingleChunkTV = value;
}
/**
 * @brief      Gets the current single frame tv.
 *
 * @return     The current single frame tv.
 */
int MainWindow::getCurrentSingleFrameTV() const
{
    return currentSingleFrameTV;
}
/**
 * @brief      Sets the current single frame tv.
 *
 * @param[in]  value  The value
 */
void MainWindow::setCurrentSingleFrameTV(int value)
{
    currentSingleFrameTV = value;
}
/**
 * @brief      Gets the current single chunk panoramic.
 *
 * @return     The current single chunk panoramic.
 */
int MainWindow::getCurrentSingleChunkPanoramic() const
{
    return currentSingleChunkPanoramic;
}
/**
 * @brief      Sets the current single chunk panoramic.
 *
 * @param[in]  value  The value
 */
void MainWindow::setCurrentSingleChunkPanoramic(int value)
{
    currentSingleChunkPanoramic = value;
}
/**
 * @brief      Gets the current single frame panoramic.
 *
 * @return     The current single frame panoramic.
 */
int MainWindow::getCurrentSingleFramePanoramic() const
{
    return currentSingleFramePanoramic;
}
/**
 * @brief      Sets the current single frame panoramic.
 *
 * @param[in]  value  The value
 */
void MainWindow::setCurrentSingleFramePanoramic(int value)
{
    currentSingleFramePanoramic = value;
}
/**
 * @brief      Gets the current begining frame panoramic.
 *
 * @return     The current begining frame panoramic.
 */
int MainWindow::getCurrentBeginingFramePanoramic() const
{
    return currentBeginingFramePanoramic;
}
/**
 * @brief      Sets the current begining frame panoramic.
 *
 * @param[in]  value  The value
 */
void MainWindow::setCurrentBeginingFramePanoramic(int value)
{
    currentBeginingFramePanoramic = value;
}
/**
 * @brief      Gets the current begining frame tv.
 *
 * @return     The current begining frame tv.
 */
int MainWindow::getCurrentBeginingFrameTV() const
{
    return currentBeginingFrameTV;
}
/**
 * @brief      Sets the current begining frame tv.
 *
 * @param[in]  value  The value
 */
void MainWindow::setCurrentBeginingFrameTV(int value)
{
    currentBeginingFrameTV = value;
}
/**
 * @brief      Gets the flag ubication defined.
 *
 * @return     The flag ubication defined.
 */
bool MainWindow::getFlagUbicationDefined() const
{
    return flagUbicationDefined;
}
/**
 * @brief      Sets the flag ubication defined.
 *
 * @param[in]  value  The value
 */
void MainWindow::setFlagUbicationDefined(bool value)
{
    flagUbicationDefined = value;
}
/**
 * @brief      Gets the flag python.
 *
 * @return     The flag python.
 */
bool MainWindow::getFlagPython() const
{
    return flagPython;
}
/**
 * @brief      Sets the flag python.
 *
 * @param[in]  value  The value
 */
void MainWindow::setFlagPython(bool value)
{
    flagPython = value;
}
/**
 * @brief Update widget where video from panoramic is shown.
 */
void MainWindow::updateVideoLabelPanoramic(){
    cv::Mat src;
    float tempScale = .55;
    src = this->panoramicFrame.clone();
    if(src.size().width > 0){
        tempScale = (float)this->ptrPanoramicVideoLabel->geometry().width()/(float)src.size().width;
        cv::resize(src,src,cv::Size(), tempScale, tempScale,cv::INTER_CUBIC);
        cvtColor(src,src,CV_BGR2RGB);
        QImage dest((const uchar *) src.data, src.cols, src.rows, src.step, QImage::Format_RGB888);
        dest.bits();
        //this->mtx.lock();
        this->ptrPanoramicVideoLabel->setPixmap(QPixmap::fromImage(dest));
        //this->mtx.unlock();
        //ui->labelPanoramic->setSrcOriginal(QPixmap::fromImage(dest));
        //this->listCameraLabels[i]->setPixmap(QPixmap::fromImage(dest));
        this->mtx.lock();
        QApplication::processEvents();
        this->mtx.unlock();
    }
}

/**
 * @brief      Update widget where video from TV broadcasting is shown.
 */
void MainWindow::updateVideoLabelTV(){
    cv::Mat src;
    float tempScale = .55;
    src = this->tvFrame.clone();
    if(src.size().width > 0){
        tempScale = (float)this->ptrTVVideoLabel->geometry().width()/(float)src.size().width;
        cv::resize(src,src,cv::Size(), tempScale, tempScale,cv::INTER_CUBIC);
        cvtColor(src,src,CV_BGR2RGB);
        QImage dest((const uchar *) src.data, src.cols, src.rows, src.step, QImage::Format_RGB888);
        dest.bits();
        //this->mtx.lock();
        this->ptrTVVideoLabel->setPixmap(QPixmap::fromImage(dest));
        //this->mtx.unlock();
        //ui->labelPanoramic->setSrcOriginal(QPixmap::fromImage(dest));
        //this->listCameraLabels[i]->setPixmap(QPixmap::fromImage(dest));
        //QApplication::processEvents();
        //this->updateAll();
    }
}


/**
 * @brief      Update all widgets on main form.
 */
void MainWindow::updateGUI(){
    /*if(this->getFlagPanoramicGUI()){
        ui->labelPanoramic->setHidden(false);
        ui->checkBoxPanoramicVideo->setHidden(false);
        ui->checkBoxPanoramicStreaming->setHidden(false);
        ui->groupBoxPanoramic->setHidden(false);
        ui->labelPanoramicFpsIn->setHidden(false);
        ui->spinBoxPanoramicFpsIn->setHidden(false);
        ui->pushButtonConnectPanoramic->setHidden(false);
        ui->checkBoxPanoramicUploadYes->setHidden(false);
        ui->checkBoxPanoramicUploadNo->setHidden(false);
        ui->groupBoxPanoramicSubir->setHidden(false);
        //!< ui->comboBoxPanoramicResolution->setHidden(false); //Por el momento esta deshabilitado de la aplicación
        //!< ui->groupBoxPanoramicResolution->setHidden(false); //Por el momento esta deshabilitado de la aplicación
        ui->labelPanoramicFpsOut->setHidden(false);
        ui->spinBoxPanoramicFpsOut->setHidden(false);
        ui->pushButtonSavePanoramic->setHidden(false);
        //!< ui->pushButtonRegisterPanoramic->setHidden(false); //Por el momento esta eshabilitado de la aplicación
        //!< ui->comboBoxRegisterPanoramic->setHidden(false); //Por el momento esta eshabilitado de la aplicación
        ui->pushButtonStopPanoramic->setHidden(false);
        ui->pushButtonReconnectPanoramic->setHidden(false);
        ui->pushButtonReStartRecordingPanoramic->setHidden(false);
        ui->lblURL->setHidden(false);
        ui->edtURL->setHidden(false);
        this->ptrPanoramicVideoLabel->setHidden(false);
    }else{
        ui->labelPanoramic->setHidden(true);
        ui->checkBoxPanoramicVideo->setHidden(true);
        ui->checkBoxPanoramicStreaming->setHidden(true);
        ui->groupBoxPanoramic->setHidden(true);
        ui->labelPanoramicFpsIn->setHidden(true);
        ui->spinBoxPanoramicFpsIn->setHidden(true);
        ui->pushButtonConnectPanoramic->setHidden(true);
        ui->checkBoxPanoramicUploadYes->setHidden(true);
        ui->checkBoxPanoramicUploadNo->setHidden(true);
        ui->groupBoxPanoramicSubir->setHidden(true);
        //!< ui->comboBoxPanoramicResolution->setHidden(true); //Por el momento esta deshabilitado de la aplicación
        //!< ui->groupBoxPanoramicResolution->setHidden(true); //Por el momento esta deshabilitado de la aplicación
        ui->labelPanoramicFpsOut->setHidden(true);
        ui->spinBoxPanoramicFpsOut->setHidden(true);
        ui->pushButtonSavePanoramic->setHidden(true);
        //!< ui->pushButtonRegisterPanoramic->setHidden(true); //Por el momento esta eshabilitado de la aplicación
        //!< ui->comboBoxRegisterPanoramic->setHidden(true); //Por el momento esta eshabilitado de la aplicación
        ui->pushButtonStopPanoramic->setHidden(true);
        ui->pushButtonReconnectPanoramic->setHidden(true);
        ui->pushButtonReStartRecordingPanoramic->setHidden(true);
        ui->lblURL->setHidden(true);
        ui->edtURL->setHidden(true);
        this->ptrPanoramicVideoLabel->setHidden(true);
    }
    if(this->getFlagTVGUI()){
        ui->labelTV->setHidden(false);
        ui->checkBoxTVVideo->setHidden(false);
        ui->checkBoxTVCapturadora->setHidden(false);
        ui->groupBoxTV->setHidden(false);
        ui->labelTVFpsIn->setHidden(false);
        ui->spinBoxTVFpsIn->setHidden(false);
        ui->pushButtonConnectTV->setHidden(false);
        ui->checkBoxTVUploadYes->setHidden(false);
        ui->checkBoxTVUploadNo->setHidden(false);
        ui->groupBoxTVSubir->setHidden(false);
        //!< ui->comboBoxTVResolution->setHidden(false); //Por el momento esta deshabilitado de la aplicación
        //!< ui->groupBoxTVResolution->setHidden(false); //Por el momento esta deshabilitado de la aplicación
        ui->labelTVFpsOut->setHidden(false);
        ui->spinBoxTVFpsOut->setHidden(false);
        ui->pushButtonSaveTV->setHidden(false);
        //!< ui->pushButtonRegisterTV->setHidden(false); //Por el momento esta eshabilitado de la aplicación
        //!< ui->comboBoxRegisterTV->setHidden(false); //Por el momento esta eshabilitado de la aplicación
        ui->pushButtonStopTV->setHidden(false);
        ui->pushButtonReconnectTV->setHidden(false);
        ui->pushButtonReStartRecordingTV->setHidden(false);
        this->ptrTVVideoLabel->setHidden(false);
    }else{
        ui->labelTV->setHidden(true);
        ui->checkBoxTVVideo->setHidden(true);
        ui->checkBoxTVCapturadora->setHidden(true);
        ui->groupBoxTV->setHidden(true);
        ui->labelTVFpsIn->setHidden(true);
        ui->spinBoxTVFpsIn->setHidden(true);
        ui->pushButtonConnectTV->setHidden(true);
        ui->checkBoxTVUploadYes->setHidden(true);
        ui->checkBoxTVUploadNo->setHidden(true);
        ui->groupBoxTVSubir->setHidden(true);
        //!< ui->comboBoxTVResolution->setHidden(true); //Por el momento esta deshabilitado de la aplicación
        //!< ui->groupBoxTVResolution->setHidden(true); //Por el momento esta deshabilitado de la aplicación
        ui->labelTVFpsOut->setHidden(true);
        ui->spinBoxTVFpsOut->setHidden(true);
        ui->pushButtonSaveTV->setHidden(true);
        //!< ui->pushButtonRegisterTV->setHidden(true); //Por el momento esta eshabilitado de la aplicación
        //!< ui->comboBoxRegisterTV->setHidden(true); //Por el momento esta eshabilitado de la aplicación
        ui->pushButtonStopTV->setHidden(true);
        ui->pushButtonReconnectTV->setHidden(true);
        ui->pushButtonReStartRecordingTV->setHidden(true);
        this->ptrTVVideoLabel->setHidden(true);
    }*/
    int height;
    int width;
    height = 0;
    width = 0;
    /*if(this->getFlagPanoramicGUI()){
        width  = 1520;
        //height = 480+240+100;
    }else{
        width  = 880;
    }*/
    //height = 480+100+50;
    width  = 1520;
    height = 480+180;
    this->setMinimumWidth(width);
    this->setMinimumHeight(height);
    this->setMaximumWidth(width);
    this->setMaximumHeight(height);
//    this->updateLabelCurrentOutputPath();
//    this->updateBeginingElementsView();
}
/**
 * @brief  Test runPythonScript
 */
void MainWindow::on_pushButton_clicked()
{
    char path[200];
    char fileName[50];
    char extension[10];
    strcpy(path,"/home/pepe/DATOS/imagenes/MuestrasPlayers2/HumanosTest");
    strcpy(fileName,"GameConfig_107");
    strcpy(extension,"jpg");
    //this->runPythonScript(path,fileName,extension);
    printf("HOLA %%");
}
/**
 * @brief      Revisando funcionamiento de SVM
 */
void MainWindow::on_pushButton_2_clicked(){
    int labels[4] = {1, -1, -1, -1};
    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, 2, CV_32F, trainingData);
    Mat labelsMat(4, 1, CV_32SC1, labels);
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);
    Vec3b green(0,255,0), blue(255,0,0);
    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = svm->predict(sampleMat);
            if (response == 1){
                image.at<Vec3b>(i,j)  = green;
            }else if (response == -1){
                image.at<Vec3b>(i,j)  = blue;
            }
        }
    }
    int thickness = -1;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness );
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness );
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness );
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness );
    thickness = 2;
    Mat sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; i++){
        const float* v = sv.ptr<float>(i);
        circle(image,  Point( (int) v[0], (int) v[1]), 6, Scalar(128, 128, 128), thickness);
    }
    //imwrite("result.png", image);        // save the image
    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);
}

//def hoggify(path,extension,is_color):
//    data=[]
//    lista=glob.glob(os.path.join(path,"*"+extension))
//    lista=np.squeeze(lista)
//    for file in lista:
//        image = cv2.cv2.imread(file, is_color)
//        dim = 128
//        img = cv2.cv2.resize(image, (dim,dim), interpolation = cv2.cv2.INTER_AREA)
//        img = hogcv2.compute(img)
//        img = np.squeeze(img)
//        data.append(img)
//    return data

/**
 * @brief      Function less than.
 *
 * @param[in]  v1    The v 1
 * @param[in]  v2    The v 2
 *
 * @return     Return result of comparison between param 1 and param 2.
 */
bool lessThan( const QString& v1, const QString& v2 )
{
    bool result = false;
    int n = v1.length();
    int m = v2.length();
    int N = std::min(n,m);
    for(int i = 0;i < N;++i){

    }
    return result;
}

/**
 * Saves the given descriptor vector to a file
 * @param descriptorVector the descriptor vector to save
 * @param _vectorIndices contains indices for the corresponding vector values (e.g. descriptorVector(0)=3.5f may have index 1)
 * @param fileName
 * @TODO Use _vectorIndices to write correct indices
 */
/*static void saveDescriptorVectorToFile(vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string fileName) {
    printf("Saving descriptor vector to file '%s'\n", fileName.c_str());
    string separator = " "; // Use blank as default separator between single features
    fstream File;
    float percent;
    File.open(fileName.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        printf("Saving %lu descriptor vector features:\t", descriptorVector.size());
        storeCursor();
        for (int feature = 0; feature < descriptorVector.size(); ++feature) {
            if ((feature % 10 == 0) || (feature == (descriptorVector.size()-1)) ) {
                percent = ((1 + feature) * 100 / descriptorVector.size());
                printf("%4u (%3.0f%%)", feature, percent);
                fflush(stdout);
                resetCursor();
            }
            File << descriptorVector.at(feature) << separator;
        }
        printf("\n");
        File << endl;
        File.flush();
        File.close();
    }
}*/
/**
 * @brief      Test of Hog on Cuda
 */
void MainWindow::on_pushButton_3_clicked(){
    int contador = 0;
    unsigned long int elapsed_nanoseconds1 = 0;
    unsigned long int elapsed_nanoseconds2 = 0;
    unsigned long int elapsed_nanoseconds3 = 0;
    double mean_elapsed_nanoseconds1 = 0;
    double mean_elapsed_nanoseconds2 = 0;
    double mean_elapsed_nanoseconds3 = 0;
    std::chrono::duration<double, std::nano> elapsed;
//    QString fileDir;
//    fileDir = QFileDialog::getExistingDirectory(this,QString::fromStdString("Selecciona folder mues"),QDir::currentPath(),QFileDialog::ShowDirsOnly);
//    this->setCurrentInputPath(fileDir.toStdString());
    QDir directory("/home/pepe/DATOS/imagenes/MuestrasPlayers4/Humanos");
    QStringList images = directory.entryList(this->qslValidExtensions,QDir::Files);
    //qSort( images.begin(), images.end(), lessThan );
    /* CUDA HOG Descriptor */
    /* Change Size() parameters to accomodate to the used image */
    Ptr<cuda::HOG> hog = cuda::HOG::create(Size(128, 128),  /* winSize */
                                           Size(16, 16),    /* blockSize */
                                           Size(8, 8),      /* blockStride */
                                           Size(8, 8),      /* cellSize */
                                           9);              /* nbins */
    HOGDescriptor hogCPU;
    //hogCPU.winSize = Size(320, 240);
    hogCPU.winSize = Size(128, 128);
    /* CUDA Stream and GpuMat */
    cuda::Stream stream;
    cuda::GpuMat cuda_src;
    cuda::GpuMat cuda_src_alpha;
    cuda::GpuMat cuda_descriptors;
    Mat          descriptors;
    Mat          img_src;
    Mat          img_gray;
    std::vector<float> descriptorsCPU;
    auto startChrono = std::chrono::steady_clock::now();
    auto endChrono = std::chrono::steady_clock::now();
    for(int i=1;i< 100;i++)
    foreach(QString filename, images) {
        QString srcFileName = directory.absolutePath()+QDir::toNativeSeparators(QString::fromStdString("/")) + filename;
        //std::cout << srcFileName.toStdString() << std::endl;
        img_src = imread(srcFileName.toStdString().c_str(), IMREAD_COLOR);
        //HOGDescriptor::compute(img_src,descriptorsCPU);
        startChrono = std::chrono::steady_clock::now();
        /* CPU */
        //std::cout << "Running HOG on CPU" << std::endl;
        cvtColor(img_src, img_gray, COLOR_BGR2GRAY);
        cv::resize(img_gray,img_gray,cv::Size(128,128));
        hogCPU.compute(img_gray, descriptorsCPU, Size(8,8), Size(0,0));
        endChrono = std::chrono::steady_clock::now();
        elapsed = endChrono-startChrono;
        elapsed_nanoseconds1 = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();

        startChrono = std::chrono::steady_clock::now();
        /* Uploading src to GPU using default stream */
        //std::cout << "Running HOG on deafult Stream" << std::endl;
        cuda_src.upload(img_src);
        cuda::cvtColor(cuda_src, cuda_src_alpha, COLOR_BGR2BGRA, 4);
        cuda::resize(cuda_src_alpha,cuda_src_alpha,cv::Size(128,128));
        hog->compute(cuda_src_alpha, cuda_descriptors);
        cuda_descriptors.download(descriptors);
        endChrono = std::chrono::steady_clock::now();
        elapsed = endChrono-startChrono;
        elapsed_nanoseconds2 = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();

        startChrono = std::chrono::steady_clock::now();
        /* Uploading src to GPU using non-default Stream */
        //std::cout << "Running HOG on non-deafult Stream" << std::endl;
        cuda_src.upload(img_src, stream);
        cuda::cvtColor(cuda_src, cuda_src_alpha, COLOR_BGR2BGRA, 4, stream);
        cuda::resize(cuda_src_alpha,cuda_src_alpha,cv::Size(128,128));
        hog->compute(cuda_src_alpha, cuda_descriptors, stream);
        cuda_descriptors.download(descriptors, stream);
        endChrono = std::chrono::steady_clock::now();
        elapsed = endChrono-startChrono;
        elapsed_nanoseconds3 = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();

        contador++;
        mean_elapsed_nanoseconds1+= elapsed_nanoseconds1;
        mean_elapsed_nanoseconds2+= elapsed_nanoseconds2;
        mean_elapsed_nanoseconds3+= elapsed_nanoseconds3;
        //std::cout << elapsed_nanoseconds1 << " " << elapsed_nanoseconds2 << " " << elapsed_nanoseconds3 << " " << std::endl;
    }
    std::cout << contador << " " << mean_elapsed_nanoseconds1 << " " << mean_elapsed_nanoseconds2 << " " << mean_elapsed_nanoseconds3 << " " << std::endl;
    if(contador > 0){
        mean_elapsed_nanoseconds1/=(double)contador;
        mean_elapsed_nanoseconds2/=(double)contador;
        mean_elapsed_nanoseconds3/=(double)contador;
    }
    std::cout << mean_elapsed_nanoseconds1 << " " << mean_elapsed_nanoseconds2 << " " << mean_elapsed_nanoseconds3 << " " << std::endl;


}
/**
 * @brief      Gets the files in directory.
 *
 * @param      Files  The files
 *
 * @return     The files in directory.
 */
int MainWindow::getFilesInDirectory(QStringList &Files){
    int n = 0;
    Files.clear();
    Files = this->directoryIn.entryList(this->qslValidExtensions,QDir::Files);
    n = Files.length();
    return n;
}
/**
 * @brief      Gets the files in directory.
 *
 * @param[in]  directory  The directory
 * @param      Files      The files
 *
 * @return     The files in directory.
 */
int MainWindow::getFilesInDirectory(QDir directory, QStringList &Files){
    int n = 0;
    Files.clear();
    Files = directory.entryList(qslValidExtensions,QDir::Files);
    n = Files.length();
    for(int i = 0; i < n;i++){
        QString srcFileName = directory.absolutePath()+QDir::toNativeSeparators(QString::fromStdString("/")) + Files.at(i);
        Files[i] = srcFileName;
    }
    return n;
}
/**
 * @brief      Calculates the features from input.
 *
 * @param[in]  imageFilename  The image filename
 * @param      featureVector  The feature vector
 * @param      hog            The hog
 */
void MainWindow::calculateFeaturesFromInput(const string imageFilename, vector<float> &featureVector, HOGDescriptor& hog) {
    /** for imread flags from openCV documentation,
     * @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat imread(const string& filename, int flags)
     * @note If you get a compile-time error complaining about following line (esp. imread),
     * you either do not have a current openCV version (>2.0)
     * or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
     */
    Mat imageData = imread(imageFilename, IMREAD_GRAYSCALE);
    if (imageData.empty()){
        featureVector.clear();
        printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
        return;
    }
    // Check for mismatching dimensions
    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
        //featureVector.clear();
        printf("Warn: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
        printf(" Image will be resized\n");
        cv::resize(imageData,imageData,Size(128,128));
        //return;
    }
    vector<Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
    imageData.release(); // Release the image again after features are extracted
}
/**
 * @brief      Calculates the features from input.
 *
 * @param[in]  imageFilename  The image filename
 * @param      featureVector  The feature vector
 * @param      hog            The hog
 */
void MainWindow::calculateFeaturesFromInput(cv::Mat imageData, vector<float> &featureVector, HOGDescriptor& hog) {
    /** for imread flags from openCV documentation,
     * @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat imread(const string& filename, int flags)
     * @note If you get a compile-time error complaining about following line (esp. imread),
     * you either do not have a current openCV version (>2.0)
     * or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
     */
    if (imageData.empty()){
        featureVector.clear();
        printf("Error: HOG image '%s' is empty, features calculation skipped!\n", "imageFilename");
        return;
    }
    // Check for mismatching dimensions
    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
        //featureVector.clear();
        printf("Warn: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!", "imageFilename", imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
        printf(" Image will be resized\n");
        cv::resize(imageData,imageData,Size(128,128));
        //return;
    }
    vector<Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
    imageData.release(); // Release the image again after features are extracted
}
/**
 * @brief      Create lower version of string.
 *
 * @param[in]  in    { parameter_description }
 *
 * @return     { description_of_the_return_value }
 */
static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}
/**
 * @brief      Stores a cursor.
 */
static void storeCursor(void) {
    printf("\033[s");
}
/**
 * @brief      Reset a cursor.
 */
static void resetCursor(void) {
    printf("\033[u");
}

/**
 * @brief  Saves the given descriptor vector to a file
 * @param descriptorVector the descriptor vector to save
 * @param _vectorIndices contains indices for the corresponding vector values (e.g. descriptorVector(0)=3.5f may have index 1)
 * @param fileName
 * @TODO Use _vectorIndices to write correct indices
 */
static void saveDescriptorVectorToFile(vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string fileName) {
    // COMENTAR para aplicacion
    //printf("Saving descriptor vector to file '%s'\n", fileName.c_str());
    string separator = " "; // Use blank as default separator between single features
    fstream File;
    float percent;
    File.open(fileName.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        // COMENTAR para aplicacion
        //printf("Saving %lu descriptor vector features:\t", descriptorVector.size());
        storeCursor();
        for (int feature = 0; feature < descriptorVector.size(); ++feature) {
            //COMENTAR para aplicacion
//            if ((feature % 10 == 0) || (feature == (descriptorVector.size()-1)) ) {
//                percent = ((1 + feature) * 100 / descriptorVector.size());
//                printf("%4u (%3.0f%%)", feature, percent);
//                fflush(stdout);
//                resetCursor();
//            }
            File << descriptorVector.at(feature) << separator;
        }
        //printf("\n");
        File << endl;
        File.flush();
        File.close();
    }
}

/**
 * @brief Test the trained detector against the same training set to get an approximate idea of the detector.
 * Warning: This does not allow any statement about detection quality, as the detector might be overfitting.
 * Detector quality must be determined using an independent test set.
 * @param hog
 */
void MainWindow::detectTrainingSetTest(const HOGDescriptor& hog, const double hitThreshold, QStringList &posFileNames, QStringList &negFileNames){
    unsigned int truePositives = 0;
    unsigned int trueNegatives = 0;
    unsigned int falsePositives = 0;
    unsigned int falseNegatives = 0;
    vector<Point> foundDetection;
    // Walk over positive training samples, generate images and detect
    //for (vector<string>::const_iterator posTrainingIterator = posFileNames.begin(); posTrainingIterator != posFileNames.end(); ++posTrainingIterator) {
    foreach(QString filename, posFileNames){
        Mat imageData = imread(filename.toStdString().c_str(), IMREAD_GRAYSCALE);
        cv::resize(imageData,imageData,Size(128,128));
        hog.detect(imageData, foundDetection, hitThreshold, winStride, trainingPadding);
        if (foundDetection.size() > 0) {
            ++truePositives;
            falseNegatives += foundDetection.size() - 1;
        } else {
            ++falseNegatives;
        }
    }
    // Walk over negative training samples, generate images and detect
    //for (vector<string>::const_iterator negTrainingIterator = negFileNames.begin(); negTrainingIterator != negFileNames.end(); ++negTrainingIterator) {
    foreach(QString filename, negFileNames){
        Mat imageData = imread(filename.toStdString().c_str(), IMREAD_GRAYSCALE);
        cv::resize(imageData,imageData,Size(128,128));
        hog.detect(imageData, foundDetection, hitThreshold, winStride, trainingPadding);
        if (foundDetection.size() > 0) {
            falsePositives += foundDetection.size();
        } else {
            ++trueNegatives;
        }
    }
    std::cout << "FINISH" << std::endl;
    printf("Results:\n\tTrue Positives: %u\n\tTrue Negatives: %u\n\tFalse Positives: %u\n\tFalse Negatives: %u\n", truePositives, trueNegatives, falsePositives, falseNegatives);
}


/**
 * @brief Shows the detections in the image
 * @param found vector containing valid detection rectangles
 * @param imageData the image in which the detections are drawn
 */
static void showDetections(const vector<Rect>& found, Mat& imageData) {
    vector<Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
        rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
    }
}
/**
 * @brief      Detect Test
 *
 * @param[in]  hog           The hog
 * @param[in]  hitThreshold  The hit threshold
 * @param      imageData     The image data
 */
void MainWindow::detectTest(const HOGDescriptor &hog, const double hitThreshold, Mat &imageData) {
    vector<Rect> found;
    Size padding(Size(8, 8));
    Size winStride(Size(8, 8));
    hog.detectMultiScale(imageData, found, hitThreshold, winStride, padding);
    showDetections(found, imageData);
}

//void MainWindow::trainSVMFromFeaturesFiles(){

//}

/**
 * @brief      SVM train test
 */
void MainWindow::on_pushButton_4_clicked(){
    LIBSVM SVM_instancia;
    HOGDescriptor hog;
    hog.winSize = Size(128, 128);
    float percent;
    this->setDirectoryIn(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/Humanos"));
    this->setDirectoryTrainSet1(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/HumanosTrain"));
    this->setDirectoryTrainSet2(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/NoHumanosTrain"));
    this->setDirectoryTestSet1(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/HumanosTest"));
    this->setDirectoryTestSet2(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/NoHumanosTest"));
    QStringList positiveTrainingImages;
    QStringList negativeTrainingImages;
    QStringList positiveTestImages;
    QStringList negativeTestImages;
    int nPositiveTrainingImages = this->getFilesInDirectory(this->directoryTrainSet1,positiveTrainingImages);
    int nNegativeTrainingImages = this->getFilesInDirectory(this->directoryTrainSet2,negativeTrainingImages);
    int nPositiveTestImages = this->getFilesInDirectory(this->directoryTestSet1,positiveTestImages);
    int nNegativeTestImages = this->getFilesInDirectory(this->directoryTestSet2,negativeTestImages);
    //foreach(QString filename, positiveTrainingImages) {
    //    std::cout << filename.toStdString() << std::endl;
    //}
    std::cout << nPositiveTrainingImages << ",";
    std::cout << nNegativeTrainingImages << ",";
    std::cout << nPositiveTestImages << ",";
    std::cout << nNegativeTestImages << std::endl;
    string featuresFile("features.dat");
    string svmModelFile = "svm_model.dat";
    string descriptorVectorFile = "descriptorvector.dat";
    string cvHOGFile = "cvHOGClassifier.yaml";
    /*int overallSamples = nPositiveTrainingImages+nNegativeTrainingImages;
    if(overallSamples == 0){
        printf("No training sample files found, nothing to do!\n");
        return;
    }
    fstream File;
        File.open(featuresFile.c_str(), ios::out);
        if (File.good() && File.is_open()) {
            //File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << featuresFile.c_str() << endl; // Remove this line for libsvm which does not support comments
            // Iterate over sample images
            for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
                storeCursor();
                vector<float> featureVector;
                // Get positive or negative sample image file path
                QString currentImageFile = (currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile) : negativeTrainingImages.at(currentFile - positiveTrainingImages.size()));
                // Output progress
                if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                    percent = ((currentFile+1) * 100 / overallSamples);
                    printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.toStdString().c_str());
                    fflush(stdout);
                    resetCursor();
                }
                // Calculate feature vector from current image file
                calculateFeaturesFromInput(currentImageFile.toStdString(), featureVector, hog);
                if (!featureVector.empty()) {
                    // Put positive or negative sample class to file,
                    // true=positive, false=negative,
                    // and convert positive class to +1 and negative class to -1 for SVMlight
                    //
                    File << ((currentFile < positiveTrainingImages.size()) ? "+1" : "-1");
                    // Save feature vector components
                    for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
                        File << " " << (feature + 1) << ":" << featureVector.at(feature);
                    }
                    File << endl;
                }
            }
            printf("\n");
            File.flush();
            File.close();
        } else {
            printf("Error opening file '%s'!\n", featuresFile.c_str());
            return;
        }*/
    /// Proceso de entrenamiento
    SVM_instancia.read_problem(const_cast<char*> (featuresFile.c_str()));
    SVM_instancia.train();
    SVM_instancia.saveModelToFile(svmModelFile);
    /// Guardando features de HOG
    vector<float> descriptorVector;
    vector<unsigned int> descriptorVectorIndices;
    SVM_instancia.getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
    saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);
    /// save opencv hog descriptor
    const double hitThreshold = SVM_instancia.getThreshold();
    std::cout << hitThreshold << std::endl;
    // Set our custom detecting vector
    hog.setSVMDetector(descriptorVector);
    hog.save(cvHOGFile);
    ///  detect
    detectTrainingSetTest(hog, hitThreshold, positiveTrainingImages, negativeTrainingImages);

    //printf("Testing training phase using training set as test set (just to check if training is ok - no detection quality conclusion with this!)\n");
    //detectTrainingSetTest(hog, hitThreshold, positiveTrainingImages, negativeTrainingImages);
}
/**
 * @brief      SVM load model test
 */
void MainWindow::on_pushButton_5_clicked()
{
    int contador = 0;
    unsigned long int elapsed_nanoseconds1 = 0;
    unsigned long int elapsed_nanoseconds2 = 0;
    unsigned long int elapsed_nanoseconds3 = 0;
    double mean_elapsed_nanoseconds1 = 0;
    double mean_elapsed_nanoseconds2 = 0;
    double mean_elapsed_nanoseconds3 = 0;
    std::chrono::duration<double, std::nano> elapsed;
    QDir directory("/home/pepe/DATOS/imagenes/MuestrasPlayers4/Humanos");
    QStringList images = directory.entryList(this->qslValidExtensions,QDir::Files);
    Ptr<cuda::HOG> hog = cuda::HOG::create(Size(128, 128),  /* winSize */
                                           Size(16, 16),    /* blockSize */
                                           Size(8, 8),      /* blockStride */
                                           Size(8, 8),      /* cellSize */
                                           9);              /* nbins */
    cuda::Stream stream;
    cuda::GpuMat cuda_src;
    cuda::GpuMat cuda_src_alpha;
    cuda::GpuMat cuda_descriptors;
    Mat          descriptors;
    Mat          results;
    Mat          img_src;
    Mat          SVMVectors;
    Mat          SVMVectorsUncompressed;
    auto startChrono = std::chrono::steady_clock::now();
    auto endChrono = std::chrono::steady_clock::now();
    cv::Ptr<cv::ml::SVM> mSvm2;
    //mSvm2 = cv::ml::SVM::load<cv::ml::SVM>("/home/pepe/DATOS/imagenes/MuestrasPlayers4/modelo_svm_cv2.yml");
    mSvm2 = cv::ml::SVM::load("/home/pepe/DATOS/imagenes/MuestrasPlayers4/modelo_svm_cv2.yml");
    std::cout <<  "C: " << mSvm2->getC() << std::endl;
    std::cout <<  "Gamma: " << mSvm2->getGamma() << std::endl;
    std::cout <<  "KernelType: " << mSvm2->getKernelType() << std::endl;
    std::cout <<  "Degree: " << mSvm2->getDegree() << std::endl;
    std::cout <<  "Type: " << mSvm2->getType() << std::endl;
    std::cout <<  "VarCount: " << mSvm2->getVarCount() << std::endl;
    std::cout <<  "Name: " << mSvm2->getDefaultName() << std::endl;
    SVMVectors = mSvm2->getSupportVectors();
    SVMVectorsUncompressed = mSvm2->getUncompressedSupportVectors();
    std::cout <<  "SupportVectors: " << mSvm2->getUncompressedSupportVectors().size() << std::endl;
    std::cout <<  "SupportVectors: " << mSvm2->getSupportVectors().size() << std::endl;

    /*foreach(QString filename, images){
        QString srcFileName = directory.absolutePath()+QDir::toNativeSeparators(QString::fromStdString("/")) + filename;
        img_src = imread(srcFileName.toStdString().c_str(), IMREAD_COLOR);
        startChrono = std::chrono::steady_clock::now();
        cuda_src.upload(img_src, stream);
        cuda::cvtColor(cuda_src, cuda_src_alpha, COLOR_BGR2BGRA, 4, stream);
        cuda::resize(cuda_src_alpha,cuda_src_alpha,cv::Size(128,128));
        hog->compute(cuda_src_alpha, cuda_descriptors, stream);
        cuda_descriptors.download(descriptors, stream);
        //std::cout << descriptors << std::endl;
        endChrono = std::chrono::steady_clock::now();
        elapsed = endChrono-startChrono;
        elapsed_nanoseconds1 = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        contador++;
        mean_elapsed_nanoseconds1+= elapsed_nanoseconds1;
        std::cout << mSvm2->predict(descriptors) << std::endl;
    }
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    std::cout << contador << " " << mean_elapsed_nanoseconds1 << std::endl;*/
}
/**
 * @brief      SVM save trained model test
 */
void MainWindow::on_pushButton_6_clicked()
{
    LIBSVM SVM_instancia;
    HOGDescriptor hog;
    hog.winSize = Size(128, 128);
    float percent;
    this->setDirectoryIn(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/Humanos"));
    this->setDirectoryTrainSet1(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/HumanosTrain"));
    this->setDirectoryTrainSet2(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/NoHumanosTrain"));
    this->setDirectoryTestSet1(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/HumanosTest"));
    this->setDirectoryTestSet2(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/NoHumanosTest"));
    QStringList positiveTrainingImages;
    QStringList negativeTrainingImages;
    QStringList positiveTestImages;
    QStringList negativeTestImages;
    int nPositiveTrainingImages = this->getFilesInDirectory(this->directoryTrainSet1,positiveTrainingImages);
    int nNegativeTrainingImages = this->getFilesInDirectory(this->directoryTrainSet2,negativeTrainingImages);
    int nPositiveTestImages = this->getFilesInDirectory(this->directoryTestSet1,positiveTestImages);
    int nNegativeTestImages = this->getFilesInDirectory(this->directoryTestSet2,negativeTestImages);
    //foreach(QString filename, positiveTrainingImages) {
    //    std::cout << filename.toStdString() << std::endl;
    //}
    std::cout << nPositiveTrainingImages << ",";
    std::cout << nNegativeTrainingImages << ",";
    std::cout << nPositiveTestImages << ",";
    std::cout << nNegativeTestImages << std::endl;
    string featuresFile("features.dat");
    string svmModelFile = "svm_model.dat";
    string descriptorVectorFile = "descriptorvector.dat";
    string cvHOGFile = "cvHOGClassifier.yaml";
    int overallSamples = nPositiveTrainingImages+nNegativeTrainingImages;
    if(overallSamples == 0){
        printf("No training sample files found, nothing to do!\n");
        return;
    }
    fstream File;
        File.open(featuresFile.c_str(), ios::out);
        if (File.good() && File.is_open()) {
            //File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << featuresFile.c_str() << endl; // Remove this line for libsvm which does not support comments
            // Iterate over sample images
            for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
                storeCursor();
                vector<float> featureVector;
                // Get positive or negative sample image file path
                QString currentImageFile = (currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile) : negativeTrainingImages.at(currentFile - positiveTrainingImages.size()));
                // Output progress
                if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                    percent = ((currentFile+1) * 100 / overallSamples);
                    printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.toStdString().c_str());
                    fflush(stdout);
                    resetCursor();
                }
                // Calculate feature vector from current image file
                calculateFeaturesFromInput(currentImageFile.toStdString(), featureVector, hog);
                if (!featureVector.empty()) {
                    // Put positive or negative sample class to file,
                    // true=positive, false=negative,
                    // and convert positive class to +1 and negative class to -1 for SVMlight
                    //
                    File << ((currentFile < positiveTrainingImages.size()) ? "+1" : "-1");
                    // Save feature vector components
                    for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
                        File << " " << (feature + 1) << ":" << featureVector.at(feature);
                    }
                    File << endl;
                }
            }
            printf("\n");
            File.flush();
            File.close();
        } else {
            printf("Error opening file '%s'!\n", featuresFile.c_str());
            return;
        }
    /// Proceso de entrenamiento
    SVM_instancia.read_problem(const_cast<char*> (featuresFile.c_str()));
    SVM_instancia.train();
    SVM_instancia.saveModelToFile(svmModelFile);
    /// Guardando features de HOG
    vector<float> descriptorVector;
    vector<unsigned int> descriptorVectorIndices;
    SVM_instancia.getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
    saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);
    /// save opencv hog descriptor
    const double hitThreshold = SVM_instancia.getThreshold();
    std::cout << hitThreshold << std::endl;
    // Set our custom detecting vector
    hog.setSVMDetector(descriptorVector);
    hog.save(cvHOGFile);
}
/**
 * @brief      SVM predict propability test.
 */
void MainWindow::on_pushButton_7_clicked()
{
    int contador = 0;
    unsigned long int elapsed_nanoseconds1 = 0;
    unsigned long int elapsed_nanoseconds2 = 0;
    unsigned long int elapsed_useconds1 = 0;
    unsigned long int elapsed_useconds2 = 0;
    unsigned long int total_elapsed_nanoseconds1 = 0;
    unsigned long int total_elapsed_nanoseconds2 = 0;
    unsigned long int total_elapsed_useconds1 = 0;
    unsigned long int total_elapsed_useconds2 = 0;
    double mean_elapsed_nanoseconds1 = 0;
    double mean_elapsed_nanoseconds2 = 0;
    std::chrono::duration<double, std::nano> elapsed1;
    std::chrono::duration<double, std::nano> elapsed2;
    //auto startChrono = std::chrono::steady_clock::now();
    auto startChrono = std::chrono::system_clock::now();
    //auto endChrono = std::chrono::steady_clock::now();
    auto endChrono = std::chrono::system_clock::now();
    LIBSVM SVM_instancia;
    HOGDescriptor hog;
    hog.winSize = Size(128, 128);
    float percent;
    this->setDirectoryIn(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/Humanos"));
    this->setDirectoryTrainSet1(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/HumanosTrain"));
    this->setDirectoryTrainSet2(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/NoHumanosTrain"));
    this->setDirectoryTestSet1(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/HumanosTest"));
    this->setDirectoryTestSet2(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/NoHumanosTest"));
    QStringList positiveTrainingImages;
    QStringList negativeTrainingImages;
    QStringList positiveTestImages;
    QStringList negativeTestImages;
    int nPositiveTrainingImages = this->getFilesInDirectory(this->directoryTrainSet1,positiveTrainingImages);
    int nNegativeTrainingImages = this->getFilesInDirectory(this->directoryTrainSet2,negativeTrainingImages);
    int nPositiveTestImages = this->getFilesInDirectory(this->directoryTestSet1,positiveTestImages);
    int nNegativeTestImages = this->getFilesInDirectory(this->directoryTestSet2,negativeTestImages);
    //foreach(QString filename, positiveTrainingImages) {
    //    std::cout << filename.toStdString() << std::endl;
    //}
    std::cout << nPositiveTrainingImages << ",";
    std::cout << nNegativeTrainingImages << ",";
    std::cout << nPositiveTestImages << ",";
    std::cout << nNegativeTestImages << std::endl;
    string featuresFile("features.dat");
    string svmModelFile = "svm_model.dat";
    string descriptorVectorFile = "descriptorvector.dat";
    string cvHOGFile = "cvHOGClassifier.yaml";
    int overallSamples = nPositiveTrainingImages+nNegativeTrainingImages;
    if(overallSamples == 0){
        printf("No training sample files found, nothing to do!\n");
        return;
    }
    SVM_instancia.loadModelFromFile(svmModelFile);
    // Guardando features de HOG
    vector<float> descriptorVector;
    vector<unsigned int> descriptorVectorIndices;
    SVM_instancia.getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
    //saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);
    /// save opencv hog descriptor
    const double hitThreshold = SVM_instancia.getThreshold();
    std::cout << hitThreshold << std::endl;
    /// Prediction
    vector<float> featureVector;
    for(int i = 0 ;i < positiveTestImages.size();++i){
        QString currentImageFile = positiveTestImages.at(i);
        //QString currentImageFile = negativeTestImages.at(0);
        //startChrono = std::chrono::steady_clock::now();
        startChrono = std::chrono::system_clock::now();
        calculateFeaturesFromInput(currentImageFile.toStdString(), featureVector, hog);
        //usleep(100000);
        //endChrono = std::chrono::steady_clock::now();
        endChrono = std::chrono::system_clock::now();
        elapsed1 = endChrono-startChrono;
        elapsed_nanoseconds1 = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed1).count();
        elapsed_useconds1 = std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count();
        total_elapsed_nanoseconds1 += elapsed_nanoseconds1;
        total_elapsed_useconds1 += elapsed_useconds1;
        //std::cout << "Tiempo Calculo Features: " << elapsed_nanoseconds1 << std::endl;
        this->x = (struct svm_node *)malloc(featureVector.size()*sizeof(struct svm_node));
        startChrono = std::chrono::system_clock::now();
        for(int j=0;j < featureVector.size();++j){
            this->x[j].value = featureVector[j];
            this->x[j].index = j+1;
        }
        double valueProbEstimate = 0;
        int classification = 0;
        classification = SVM_instancia.predictLabel(this->x,&valueProbEstimate);
        endChrono = std::chrono::system_clock::now();
        elapsed2 = endChrono-startChrono;
        elapsed_nanoseconds2 = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed2).count();
        elapsed_useconds2 = std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count();
        total_elapsed_nanoseconds2 += elapsed_nanoseconds2;
        total_elapsed_useconds2 += elapsed_useconds2;
        free(this->x);
        std::cout << "Corrida" << i << ": " <<  classification << std::endl;
        std::cout << "Probabilidad: " << valueProbEstimate << std::endl;
        //std::cout << "Tiempo Calculo Features: " << elapsed_nanoseconds1 << std::endl;
        //std::cout << "Tiempo Calculo SVM: " << elapsed_nanoseconds2 << std::endl;
        contador++;
    }
    std::cout << "Tiempo Calculo Features: " << total_elapsed_useconds1/(double)contador << std::endl;
    std::cout << "Tiempo Calculo SVM: " << total_elapsed_useconds2/(double)contador  << std::endl;
    fflush(stdout);
}

double MainWindow::computeProbability(cv::Mat image,LIBSVM &SVM_instancia,HOGDescriptor &hog,int &classification){
    double probabilidad = 0.0;
    classification = 0;
    const double hitThreshold = SVM_instancia.getThreshold();
    vector<float> featureVector;
    calculateFeaturesFromInput(image, featureVector, hog);
    struct svm_node *x;
    x = (struct svm_node *)malloc(featureVector.size()*sizeof(struct svm_node));
    for(int j=0;j < featureVector.size();++j){
        x[j].value = featureVector[j];
        x[j].index = j+1;
    }
    classification = SVM_instancia.predictLabel(x,&probabilidad);
    free(x);
    return probabilidad;
}

/**
 * @brief      SVM cuda version of predict propability test.
 */
void MainWindow::on_pushButton_8_clicked()
{
    LIBSVM SVM_instancia;
    HOGDescriptor hog;
    hog.winSize = Size(128, 128);
    string svmModelFile = "svm_model.dat";
    SVM_instancia.loadModelFromFile(svmModelFile);
    int classification = 0;
    double probabilidad = 0.0;
    float percent;
    this->setDirectoryIn(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/Humanos"));
    this->setDirectoryTrainSet1(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/HumanosTrain"));
    this->setDirectoryTrainSet2(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/NoHumanosTrain"));
    this->setDirectoryTestSet1(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/HumanosTest"));
    this->setDirectoryTestSet2(QDir("/home/pepe/DATOS/imagenes/MuestrasPlayers4/NoHumanosTest"));
    QStringList positiveTrainingImages;
    QStringList negativeTrainingImages;
    QStringList positiveTestImages;
    QStringList negativeTestImages;
    int nPositiveTrainingImages = this->getFilesInDirectory(this->directoryTrainSet1,positiveTrainingImages);
    int nNegativeTrainingImages = this->getFilesInDirectory(this->directoryTrainSet2,negativeTrainingImages);
    int nPositiveTestImages = this->getFilesInDirectory(this->directoryTestSet1,positiveTestImages);
    int nNegativeTestImages = this->getFilesInDirectory(this->directoryTestSet2,negativeTestImages);
    int overallSamples = nPositiveTrainingImages+nNegativeTrainingImages;
    if(overallSamples == 0){
        printf("No training sample files found, nothing to do!\n");
        return;
    }

    /// Prediction
    for(int i = 0 ;i < positiveTestImages.size();++i){
        QString currentImageFile = positiveTestImages.at(i);
        Mat imageData = cv::imread(currentImageFile.toStdString(), IMREAD_GRAYSCALE);
        probabilidad = this->computeProbability(imageData,SVM_instancia,hog,classification);
        std::cout << probabilidad << std::endl;

    }
    fflush(stdout);
}
