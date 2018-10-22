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
}

MainWindow::~MainWindow()
{
    delete ui;
}

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

bool MainWindow::getFlagPython() const
{
    return flagPython;
}

void MainWindow::setFlagPython(bool value)
{
    flagPython = value;
}

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

void MainWindow::on_pushButton_3_clicked(){
    QString fileDir;
   //fileDir =  QFileDialog::getExistingDirectory(this, tr("Selecciona Folder de salida"), QDir::currentPath(), tr("Folders"));
   fileDir = QFileDialog::getExistingDirectory(this,QString::fromStdString("Selecciona folder de salida"),QDir::currentPath(),QFileDialog::ShowDirsOnly);
   this->setCurrentOutputPath(fileDir.toStdString());
   this->setFlagUbicationDefined(true);
   this->updateLabelCurrentOutputPath();
}
