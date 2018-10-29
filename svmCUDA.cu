#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

#define NBINS 100

#define BLOCKSIZE 16

#define RANGE_H 255. // 179.
#define RANGE_L 255.
#define RANGE_S 255.



const int WINDOWS_NUMBER = 4;

const int HIST_SIZE = (NBINS * 3);

const int FEATURES_SIZE = HIST_SIZE * WINDOWS_NUMBER + 1;


#define GAUSSIAN_LENGTH 8
#define GAUSSIAN_LENGTH_W 26//2 + 6(parametros) * 4(secciones)

// 0 - class
// 1 - prob treshold
// 2 - h median
// 3 - h desv
// 4 - l median
// 5 - l desv
// 6 - s median
// 7 - s desv


int colorBytes;
int grayBytes;
int ProbBytes;

float *d_ParametersHeightForSquare, *d_ParametersWidthForSquare;

unsigned char *d_FieldImage;

unsigned char *d_PixelClass;
float *d_Probability;

unsigned char *d_PixelClass2;
float *d_Probability2;


float *d_gaussians;
int numberOfGaussians;

float *d_gaussians2;
int numberOfGaussians2;

int *d_numberOfClasses;
int *d_kPerClass;
float *d_Histogram;
float *d_maxDistances;

int *d_numberOfClasses_ball;
int *d_kPerClass_ball;
float *d_Histogram_ball;
float *d_maxDistances_ball;

int NumberOfClasses_ball;
int nHistogram_ball;

int NumberOfClasses;
int nHistogram;


int numberHistograms;
float *d_histograms;

int numberHistograms2;
float *d_histograms2;



static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if(err!=cudaSuccess)
    {
        fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
        std::cin.get();
        exit(EXIT_FAILURE);
    }
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)




__global__ void ParticleFilterNClassCUDA_kernel( unsigned char* FieldImage,
                                                 float *Histogram,
                                                 float *distances,
                                                 int nHistogram,
                                                 int *kPerClass,
                                                 int numberOfClasses,
                                                 float *ParametersHeightforSquare,
                                                 float *ParametersWidthforSquare,
                                                 int width,
                                                 int height,
                                                 int colorWidthStep,
                                                 int grayWidthStep,
                                                 int ProbWidthStep,
                                                 unsigned char* PixelClass,
                                                 float *Probability)
{


    //2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int color_tid = yIndex * colorWidthStep + (3 * xIndex);


    //Only valid threads perform memory I/O
    if(xIndex < width && yIndex < height && xIndex % 4 == 0 && yIndex % 4 == 0 &&
            !(FieldImage[color_tid] == 0 && FieldImage[color_tid + 1] == 0 && FieldImage[color_tid + 2] == 0)){

        const int gray_tid  = yIndex * width + xIndex;

        const int prob_tid  = gray_tid; //yIndex * width + xIndex;

        //Compute the window for this point
        const int HalfWindowWidth = (int)round(((((float)yIndex-ParametersWidthforSquare[2])*(ParametersWidthforSquare[1]/ParametersWidthforSquare[0]))+ParametersWidthforSquare[3])/2.0f);
        const int HalfWindowHeight = (int)round(((((float)yIndex-ParametersHeightforSquare[2])*(ParametersHeightforSquare[1]/ParametersHeightforSquare[0]))+ParametersHeightforSquare[3])/2.0);

        int HHistogramTop[NBINS];
        int LHistogramTop[NBINS];
        int SHistogramTop[NBINS];
        int HHistogrambutton[NBINS];
        int LHistogrambutton[NBINS];
        int SHistogrambutton[NBINS];

        int bytesNbins = NBINS * sizeof(int);

        memset(HHistogramTop, 0, bytesNbins);
        memset(LHistogramTop, 0, bytesNbins);
        memset(SHistogramTop, 0, bytesNbins);
        memset(HHistogrambutton, 0, bytesNbins);
        memset(LHistogrambutton, 0, bytesNbins);
        memset(SHistogrambutton, 0, bytesNbins);

        int SumaValidPixels;
        int ValidTop = 0;
        int ValidButton = 0;

        int limitii = min(yIndex + HalfWindowHeight, height - 1);
        int limitjj = min(xIndex + HalfWindowWidth, width - 1);

        for (int ii = max(yIndex - HalfWindowHeight, 0); ii < limitii; ii++){
            for (int jj = max(xIndex - HalfWindowWidth, 0); jj < limitjj; jj++){

                //Get the color image values
                const int colorIdPixel = ii * colorWidthStep + (3 * jj);
                const unsigned char h_pixel	 = FieldImage[colorIdPixel ];
                const unsigned char l_pixel = FieldImage[colorIdPixel  + 1];
                const unsigned char s_pixel	 = FieldImage[colorIdPixel  + 2];

                if (!(h_pixel == 0 && l_pixel == 0 && s_pixel == 0))
                {
                    if (ii < yIndex){
                        ValidTop++;
                        //Make the TOP histogram
                        HHistogramTop[(int)(h_pixel * NBINS/RANGE_H)]++;
                        LHistogramTop[(int)(l_pixel * NBINS/RANGE_L)]++;
                        SHistogramTop[(int)(s_pixel * NBINS/RANGE_S)]++;
                    }
                    else{
                        ValidButton++;
                        //Make the BUTTON histogram
                        HHistogrambutton[(int)(h_pixel * NBINS/RANGE_H)]++;
                        LHistogrambutton[(int)(l_pixel * NBINS/RANGE_L)]++;
                        SHistogrambutton[(int)(s_pixel * NBINS/RANGE_S)]++;
                    }
                }
            }
        }

        SumaValidPixels = ValidButton + ValidTop;

        double a = fabs((double)ValidTop/SumaValidPixels - (double)ValidButton/SumaValidPixels);
        if (true) //SumaValidPixels > HalfWindowWidth * HalfWindowHeight * 1) //&& a < .3)
        {
            //Checar si se parecen los histogramas
            float* Distance = new float[nHistogram];

            for(int n = 0; n < nHistogram; n++){
                Distance[n] = 0;
                for (int K=0;K<NBINS;K++){
                    Distance[n] += sqrtf((HHistogramTop[K]/(float)SumaValidPixels) * Histogram[HIST_SIZE * n + K]);
                    Distance[n] += sqrtf((LHistogramTop[K]/(float)SumaValidPixels) * Histogram[HIST_SIZE * n + K + (NBINS)]);
                    Distance[n] += sqrtf((SHistogramTop[K]/(float)SumaValidPixels) * Histogram[HIST_SIZE * n + K + (2 * NBINS)]);
                    Distance[n] += sqrtf((HHistogrambutton[K]/(float)SumaValidPixels) * Histogram[HIST_SIZE * n + K + (3 * NBINS)]);
                    Distance[n] += sqrtf((LHistogrambutton[K]/(float)SumaValidPixels) * Histogram[HIST_SIZE * n + K + (4 * NBINS)]);
                    Distance[n] += sqrtf((SHistogrambutton[K]/(float)SumaValidPixels) * Histogram[HIST_SIZE * n + K + (5 * NBINS)]);
                }

                //  Distance[n]=Distance[n]/((float)NBINS*6.0);

                Distance[n] = (1- (Distance[n]/6.0)) ;//* SumaValidPixels;
            }


            float minDistance = Distance[0];
            int minIndex = 0;
            for(int n = 1; n < nHistogram; n++){
                if(Distance[n] < minDistance){
                    minDistance = Distance[n];
                    minIndex = n;
                }
            }

            delete[] Distance;

            int kNum = 0;

            for (int n = 0; n < numberOfClasses; n++) {
                kNum += kPerClass[n];

                if(minIndex < kNum ) //&& minDistance) // < distances[n])
                {
                    PixelClass[gray_tid] = static_cast<unsigned char>(n + 1);
                    Probability[prob_tid] = static_cast<float>(minDistance);
                    break;
                }
            }
        }
    }
}


void ReserveCudaMemory(std::vector<std::vector<float> > Histogram, std::vector<float> maxDistances, int _nHistogram, std::vector<int> kPerClasses,
                       int SizeHistograms, int _NumberOfClasses, float *ParametersHeightForSquare, float *ParametersWidthForSquare){

    //Calculate total number of bytes of input and output image

    int ParametersForSquareBytes = 4 * sizeof(float);
    int HistogramsBytes = SizeHistograms * sizeof(float);
    NumberOfClasses =_NumberOfClasses;
    nHistogram = _nHistogram;

    float *h_histogram = new float[nHistogram*SizeHistograms];

    for(int i = 0; i < nHistogram; i++){
        for(int j = 0; j < SizeHistograms; j++){
            h_histogram[j+i*SizeHistograms] = Histogram[i][j];
        }
    }

    SAFE_CALL(cudaMalloc<int>(&d_kPerClass, sizeof(int) * kPerClasses.size()),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_Histogram, nHistogram * HistogramsBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_maxDistances, sizeof(float) * maxDistances.size()),"CUDA Malloc Failed");

    //SAFE_CALL(cudaMemcpy(&d_numberOfClasses, &NumberOfClasses, sizeof(int), cudaMemcpyHostToDevice),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_kPerClass, kPerClasses.data(), sizeof(int) * kPerClasses.size(), cudaMemcpyHostToDevice),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_Histogram, h_histogram, HistogramsBytes * nHistogram, cudaMemcpyHostToDevice),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_maxDistances, maxDistances.data(), sizeof(float) * maxDistances.size(), cudaMemcpyHostToDevice),"CUDA Malloc Failed");

    //Allocate device memory
    SAFE_CALL(cudaMalloc<float>(&d_ParametersHeightForSquare, ParametersForSquareBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_ParametersWidthForSquare, ParametersForSquareBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_ParametersHeightForSquare, ParametersHeightForSquare, ParametersForSquareBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_ParametersWidthForSquare, ParametersWidthForSquare, ParametersForSquareBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    delete[] h_histogram;

}


void ReserveCudaMemoryTexture(std::vector<std::vector<float> > histograms, std::vector<std::vector<float> > histograms2, std::vector<std::vector<float> > gaussians, std::vector<float> ParametersHeightForSquare, std::vector<float> ParametersWidthForSquare,
                              cv::Mat FieldImage, cv::Mat PixelClass, cv::Mat Probability){


    colorBytes = FieldImage.step * FieldImage.rows;
    grayBytes = PixelClass.step * PixelClass.rows;
    ProbBytes = Probability.cols * Probability.rows * sizeof(float) ;

    int ParametersForSquareBytes = 4 * sizeof(float);



    numberHistograms = histograms.size();

    std::cout<<"Resevando memoria de cuda para "<< numberHistograms <<" histogramas";

    int HistogramsBytes = numberHistograms * FEATURES_SIZE * sizeof(float);

    float *h_histogram = new float[histograms.size() * histograms[0].size()];


    for(unsigned i = 0; i < numberHistograms; i++){
        for(unsigned j = 0; j < histograms[i].size(); j++){
            h_histogram[j + i * FEATURES_SIZE] = histograms[i][j];
        }
    }



    numberHistograms2 = histograms2.size();
    std::cout<<"Resevando memoria de cuda para "<<numberHistograms2 <<" histogramas";


    int HistogramsBytes2 = numberHistograms2 * FEATURES_SIZE * sizeof(float);

    float *h_histogram2 = new float[histograms2.size() * histograms2[0].size()];


    for(unsigned i = 0; i < numberHistograms2; i++){
        for(unsigned j = 0; j < histograms2[i].size(); j++){
            h_histogram2[j + i * FEATURES_SIZE] = histograms2[i][j];
        }
    }




    numberOfGaussians = gaussians.size();

    int gaussiansSize = gaussians.size() * GAUSSIAN_LENGTH_W * sizeof(float);

    float *h_gaussians = new float[numberOfGaussians * GAUSSIAN_LENGTH_W];

    for(int i = 0; i < numberOfGaussians; i++){
        for(int j = 0; j < GAUSSIAN_LENGTH_W; j++){
            h_gaussians[j + i * GAUSSIAN_LENGTH_W] = gaussians[i][j];
        }
    }



    SAFE_CALL(cudaMalloc<float>(&d_gaussians, gaussiansSize),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_gaussians, h_gaussians, gaussiansSize, cudaMemcpyHostToDevice),"CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc<float>(&d_histograms, HistogramsBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_histograms, h_histogram, HistogramsBytes, cudaMemcpyHostToDevice),"CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc<float>(&d_histograms2, HistogramsBytes2),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_histograms2, h_histogram2, HistogramsBytes2, cudaMemcpyHostToDevice),"CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc<float>(&d_ParametersHeightForSquare, ParametersForSquareBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_ParametersWidthForSquare, ParametersForSquareBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_ParametersHeightForSquare, ParametersHeightForSquare.data(), ParametersForSquareBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_ParametersWidthForSquare, ParametersWidthForSquare.data(), ParametersForSquareBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    SAFE_CALL(cudaMalloc<unsigned char>(&d_FieldImage, colorBytes),"CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc<unsigned char>(&d_PixelClass, grayBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_Probability, ProbBytes),"CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc<unsigned char>(&d_PixelClass2, grayBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_Probability2, ProbBytes),"CUDA Malloc Failed");

    delete[] h_histogram;
    delete[] h_histogram2;
    delete[] h_gaussians;

}



void ReserveCudaMemoryBall(std::vector<std::vector<float> > Histogram, std::vector<float> maxDistances, int _nHistogram, std::vector<int> kPerClasses,
                           int SizeHistograms, int _NumberOfClasses){

    int HistogramsBytes = SizeHistograms * sizeof(float);
    NumberOfClasses_ball =_NumberOfClasses;
    nHistogram_ball = _nHistogram;

    float *h_histogram = new float[nHistogram*SizeHistograms];

    for(int i = 0; i < nHistogram; i++){
        for(int j = 0; j < SizeHistograms; j++){
            h_histogram[j+i*SizeHistograms] = Histogram[i][j];
        }
    }

    SAFE_CALL(cudaMalloc<int>(&d_kPerClass_ball, sizeof(int) * kPerClasses.size()),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_Histogram_ball, nHistogram * HistogramsBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_maxDistances_ball, sizeof(float) * maxDistances.size()),"CUDA Malloc Failed");

    SAFE_CALL(cudaMemcpy(d_kPerClass_ball, kPerClasses.data(), sizeof(int) * kPerClasses.size(), cudaMemcpyHostToDevice),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_Histogram_ball, h_histogram, HistogramsBytes * nHistogram, cudaMemcpyHostToDevice),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_maxDistances_ball, maxDistances.data(), sizeof(float) * maxDistances.size(), cudaMemcpyHostToDevice),"CUDA Malloc Failed");


    delete[] h_histogram;

}

void FreeCudaMemory(){

    //Free the device memory
    SAFE_CALL(cudaFree(d_gaussians),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_histograms),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_histograms2),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_ParametersHeightForSquare),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_ParametersWidthForSquare),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_FieldImage),"CUDA Malloc Failed");
    SAFE_CALL(cudaFree(d_PixelClass),"CUDA Malloc Failed");
    SAFE_CALL(cudaFree(d_Probability),"CUDA Malloc Failed");
    SAFE_CALL(cudaFree(d_PixelClass2),"CUDA Malloc Failed");
    SAFE_CALL(cudaFree(d_Probability2),"CUDA Malloc Failed");
 //   SAFE_CALL(cudaFree(d_kPerClass),"CUDA Malloc Failed");
 //   SAFE_CALL(cudaFree(d_maxDistances),"CUDA Malloc Failed");

}


void ParticleFilterNClassCUDA(const cv::Mat& FieldImage, cv::Mat& PixelClass, cv::Mat& Probability)
{

    //         cudaEvent_t start_hd, stop_hd;
    //         cudaEvent_t start_dh, stop_dh;
    //         cudaEvent_t start_k, stop_k;
    //         cudaEventCreate(&start_hd); cudaEventCreate(&stop_hd);
    //         cudaEventCreate(&start_dh); cudaEventCreate(&stop_dh);
    //         cudaEventCreate(&start_k); cudaEventCreate(&stop_k);



    colorBytes = FieldImage.step * FieldImage.rows;
    grayBytes = PixelClass.step * PixelClass.rows;
    ProbBytes = Probability.cols * Probability.rows * sizeof(float) ;


    //       cudaEventRecord(start_hd, 0);

    SAFE_CALL(cudaMalloc<unsigned char>(&d_FieldImage, colorBytes),"CUDA Malloc Failed");


    SAFE_CALL(cudaMalloc<unsigned char>(&d_PixelClass, grayBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_Probability, ProbBytes),"CUDA Malloc Failed");

    SAFE_CALL(cudaMemset(d_PixelClass, 0, grayBytes),"CUDA Memset Failed");

    SAFE_CALL(cudaMemcpy(d_FieldImage, FieldImage.ptr(), colorBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    //  cudaEventRecord(stop_hd, 0); cudaEventSynchronize(stop_hd);


    //Specify a reasonable block size
    const dim3 block(BLOCKSIZE ,BLOCKSIZE);
    //Calculate grid size to cover the whole image
    const dim3 grid((FieldImage.cols + block.x - 1)/block.x, (FieldImage.rows + block.y - 1)/block.y);


    //   cudaEventRecord(start_k, 0);
    //Launch the color conversion kernel
    ParticleFilterNClassCUDA_kernel<<<grid, block>>>(d_FieldImage, d_Histogram, d_maxDistances, nHistogram,
                                                     d_kPerClass, NumberOfClasses,
                                                     d_ParametersHeightForSquare,d_ParametersWidthForSquare, FieldImage.cols, FieldImage.rows, FieldImage.step,
                                                     PixelClass.step, Probability.step, d_PixelClass, d_Probability);


    //Synchronize to check for any kernel launch errors
    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
    //    cudaEventRecord(stop_k, 0); cudaEventSynchronize(stop_k);


    //     cudaEventRecord(start_dh, 0);
    //Copy back data from destination device meory to OpenCV output image
    SAFE_CALL(cudaMemcpy(PixelClass.ptr(), d_PixelClass,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(Probability.ptr(), d_Probability,ProbBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

    SAFE_CALL(cudaFree(d_FieldImage),"CUDA Free Failed");

    SAFE_CALL(cudaFree(d_PixelClass),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_Probability),"CUDA Free Failed");
    //        cudaEventRecord(stop_dh, 0); cudaEventSynchronize(stop_dh);

    //    float hostToDeviceTime, deviceToHostTime, kernelTime;
    //     cudaEventElapsedTime(&hostToDeviceTime, start_hd, stop_hd);
    //     cudaEventElapsedTime(&deviceToHostTime, start_dh, stop_dh);
    //     cudaEventElapsedTime(&kernelTime, start_k, stop_k);

    //     printf("Tiempo de copiar datos de host to device %f \n", hostToDeviceTime);
    //     printf("Tiempo de copiar datos de device to host %f \n", deviceToHostTime);
    //     printf("Tiempo de kernel %f en milisegundos\n", kernelTime);

    //       cudaEventDestroy(start_hd); cudaEventDestroy(stop_hd);
    //       cudaEventDestroy(start_dh); cudaEventDestroy(stop_dh);
    //       cudaEventDestroy(start_k); cudaEventDestroy(stop_k);


}


void ParticleFilterBallCUDA(const cv::Mat& FieldImage, cv::Mat& PixelClass, cv::Mat& Probability)
{


    colorBytes = FieldImage.step * FieldImage.rows;
    grayBytes = PixelClass.step * PixelClass.rows;
    ProbBytes = Probability.cols * Probability.rows * sizeof(float) ;

    SAFE_CALL(cudaMalloc<unsigned char>(&d_FieldImage, colorBytes),"CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc<unsigned char>(&d_PixelClass, grayBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_Probability, ProbBytes),"CUDA Malloc Failed");

    SAFE_CALL(cudaMemset(d_PixelClass, 0, grayBytes),"CUDA Memset Failed");

    SAFE_CALL(cudaMemcpy(d_FieldImage, FieldImage.ptr(), colorBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    //Specify a reasonable block size
    const dim3 block(BLOCKSIZE ,BLOCKSIZE);
    //Calculate grid size to cover the whole image
    const dim3 grid((FieldImage.cols + block.x - 1)/block.x, (FieldImage.rows + block.y - 1)/block.y);


    ParticleFilterNClassCUDA_kernel<<<grid, block>>>(d_FieldImage, d_Histogram, d_maxDistances, nHistogram,
                                                     d_kPerClass, NumberOfClasses,
                                                     d_ParametersHeightForSquare,d_ParametersWidthForSquare, FieldImage.cols, FieldImage.rows, FieldImage.step,
                                                     PixelClass.step, Probability.step, d_PixelClass, d_Probability);


    //Synchronize to check for any kernel launch errors
    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    SAFE_CALL(cudaMemcpy(PixelClass.ptr(), d_PixelClass, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(Probability.ptr(), d_Probability, ProbBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

    SAFE_CALL(cudaFree(d_FieldImage),"CUDA Free Failed");

    SAFE_CALL(cudaFree(d_PixelClass),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_Probability),"CUDA Free Failed");
}


__global__ void ParticleFilterBallCUDA_kernel( unsigned char* FieldImage,
                                               float *Histogram,
                                               float *distances,
                                               int nHistogram,
                                               int *kPerClass,
                                               int numberOfClasses,
                                               float *ParametersHeightforSquare,
                                               float *ParametersWidthforSquare,
                                               int width,
                                               int height,
                                               int colorWidthStep,
                                               int grayWidthStep,
                                               int ProbWidthStep,
                                               unsigned char* PixelClass,
                                               float *Probability)
{


    //2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int color_tid = yIndex * colorWidthStep + (3 * xIndex);


    //Only valid threads perform memory I/O
    if(xIndex < width && yIndex < height && //xIndex % 4 == 0 && yIndex % 4 == 0 &&
            FieldImage[color_tid] > 0 && FieldImage[color_tid + 1] > 0 && FieldImage[color_tid + 2] > 0){

        const int gray_tid  = yIndex * width + xIndex;

        const int prob_tid  = gray_tid; //yIndex * width + xIndex;

        //Compute the window for this point
        const int HalfWindowWidth = (int)round(((((float)yIndex-ParametersWidthforSquare[2])*(ParametersWidthforSquare[1]/ParametersWidthforSquare[0]))+ParametersWidthforSquare[3])/4.0f);
        const int HalfWindowHeight = (int)round(((((float)yIndex-ParametersHeightforSquare[2])*(ParametersHeightforSquare[1]/ParametersHeightforSquare[0]))+ParametersHeightforSquare[3])/4.0);

        int HHistogram[NBINS];
        int LHistogram[NBINS];
        int SHistogram[NBINS];

        int bytesNbins = NBINS * sizeof(int);

        memset(HHistogram, 0, bytesNbins);
        memset(LHistogram, 0, bytesNbins);
        memset(SHistogram, 0, bytesNbins);

        int SumaValidPixels;

        int limitii = min(yIndex + HalfWindowHeight, height - 1);
        int limitjj = min(xIndex + HalfWindowWidth, width - 1);

        for (int ii = max(yIndex - HalfWindowHeight, 0); ii < limitii; ii++){
            for (int jj = max(xIndex - HalfWindowWidth, 0); jj < limitjj; jj++){

                //Get the color image values
                const int colorIdPixel = ii * colorWidthStep + (3 * jj);
                const unsigned char h_pixel	 = FieldImage[colorIdPixel];
                const unsigned char l_pixel = FieldImage[colorIdPixel  + 1];
                const unsigned char s_pixel	 = FieldImage[colorIdPixel  + 2];

                if (h_pixel > 0 && l_pixel > 0 && s_pixel > 0)
                {
                    SumaValidPixels++;

                    HHistogram[(int)(h_pixel * NBINS/RANGE_H)]++;
                    LHistogram[(int)(l_pixel * NBINS/RANGE_L)]++;
                    SHistogram[(int)(s_pixel * NBINS/RANGE_S)]++;

                }
            }
        }


        float* Distance = new float[nHistogram];

        for(int n = 0; n < nHistogram; n++){
            Distance[n] = 0;
            for (int K=0;K<NBINS;K++){
                Distance[n] += sqrtf((HHistogram[K]/(float)SumaValidPixels) * Histogram[HIST_SIZE * n + K]);
                Distance[n] += sqrtf((LHistogram[K]/(float)SumaValidPixels) * Histogram[HIST_SIZE * n + K + (NBINS)]);
                Distance[n] += sqrtf((SHistogram[K]/(float)SumaValidPixels) * Histogram[HIST_SIZE * n + K + (2 * NBINS)]);

            }

            //  Distance[n]=Distance[n]/((float)NBINS*6.0);

            float Decay=1.0;///(float)SumaValidPixels;
            Distance[n]=(1- (Distance[n]/6.0))*Decay;
        }


        float minDistance = Distance[0];
        int minIndex = 0;
        for(int n = 1; n < nHistogram; n++){
            if(Distance[n] < minDistance){
                minDistance = Distance[n];
                minIndex = n;
            }
        }

        delete[] Distance;

        int kNum = 0;

        for (int n = 0; n < numberOfClasses; n++) {
            kNum += kPerClass[n];

            if(minIndex < kNum) // && minDistance < distances[n])
            {
                PixelClass[gray_tid] = static_cast<unsigned char>(n + 1);
                Probability[prob_tid] = static_cast<float>(minDistance);

                break;
            }
        }
    }
}


__global__ void ParticleFilterBayes_kernel( unsigned char* FieldImage,
                                            unsigned char* PixelClass,
                                            float *Probability,
                                            float *ParametersHeightforSquare,
                                            float *ParametersWidthforSquare,
                                            int width,
                                            int height,
                                            int colorWidthStep,
                                            float *gaussians,
                                            int gaussiansNumber
                                            )
{

    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

    if(xIndex < width && yIndex < height && //xIndex % 4 == 0 && yIndex % 4 == 0 &&
            !(FieldImage[color_tid] == 0 && FieldImage[color_tid + 1] == 0 && FieldImage[color_tid + 2] == 0)){


        const int HalfWindowWidth = (int)round(((((float)yIndex-ParametersWidthforSquare[2])*(ParametersWidthforSquare[1]/ParametersWidthforSquare[0]))+ParametersWidthforSquare[3])/2.0f);
        const int HalfWindowHeight = (int)round(((((float)yIndex-ParametersHeightforSquare[2])*(ParametersHeightforSquare[1]/ParametersHeightforSquare[0]))+ParametersHeightforSquare[3])/2.0f);

        float H = 0;
        float L = 0;
        float S = 0;
        float n = 0;
        int limityy = min(yIndex + HalfWindowHeight, height - 1);
        int limitxx = min(xIndex + HalfWindowWidth, width - 1);

        for (int yy = max(yIndex - HalfWindowHeight, 0); yy < limityy; yy++){
            for (int xx = max(xIndex - HalfWindowWidth, 0); xx < limitxx; xx++){

                const int colorIdPixel = yy * colorWidthStep + (3 * xx);
                const unsigned char h_pixel	= FieldImage[colorIdPixel];
                const unsigned char l_pixel = FieldImage[colorIdPixel + 1];
                const unsigned char s_pixel	= FieldImage[colorIdPixel + 2];

                if (!(h_pixel == 0 && l_pixel == 0 && s_pixel == 0)){
                    n++;
                    H += h_pixel;
                    L += l_pixel;
                    S += s_pixel;
                }
            }
        }


        if(n > HalfWindowHeight * HalfWindowWidth *.2){
            float percent = n * 100 / (4 * HalfWindowHeight * HalfWindowWidth) * 1000;
            H /= n;
            L /= n;
            S /= n;

            int maxIndex;
            float maxProb = 0;

            for (unsigned k = 0; k < gaussiansNumber; k++){
                int gausPos = GAUSSIAN_LENGTH * k;

                float PH = exp( (H - gaussians[gausPos + 2])*(H-gaussians[gausPos + 2]) / (-2*gaussians[gausPos + 3])) / sqrt(2* M_PI *gaussians[gausPos + 3]);
                float PL = exp( (L - gaussians[gausPos + 4])*(L-gaussians[gausPos + 4]) / (-2*gaussians[gausPos + 5])) / sqrt(2* M_PI *gaussians[gausPos + 5]);
                float PS = exp( (S - gaussians[gausPos + 6])*(S-gaussians[gausPos + 6]) / (-2*gaussians[gausPos + 7])) / sqrt(2* M_PI *gaussians[gausPos + 7]);
                float prob = PH * PL * PS;
                if(prob > maxProb){
                    maxProb = prob;
                    maxIndex = k;
                }
            }

            if(maxProb > gaussians[GAUSSIAN_LENGTH * maxIndex + 1])
            {
                const int gray_tid  = yIndex * width + xIndex;
                PixelClass[gray_tid] = static_cast<unsigned char>(gaussians[GAUSSIAN_LENGTH * maxIndex]);
                Probability[gray_tid] = static_cast<float>(maxProb*percent);
            }
        }
    }
}

__global__ void ParticleFilterWindowsBayes_kernel( unsigned char* FieldImage,
                                                   unsigned char* PixelClass,
                                                   float *Probability,
                                                   float *ParametersHeightforSquare,
                                                   float *ParametersWidthforSquare,
                                                   int width,
                                                   int height,
                                                   int colorWidthStep,
                                                   float *gaussians,
                                                   int gaussiansNumber
                                                   )
{

    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

    if(xIndex < width && yIndex < height && //xIndex % 4 == 0 && yIndex % 4 == 0 &&
            !(FieldImage[color_tid] == 0 && FieldImage[color_tid + 1] == 0 && FieldImage[color_tid + 2] == 0)){


        const int HalfWindowWidth = (int)round(((((float)yIndex-ParametersWidthforSquare[2])*(ParametersWidthforSquare[1]/ParametersWidthforSquare[0]))+ParametersWidthforSquare[3])/2.0f);
        const int HalfWindowHeight = (int)round(((((float)yIndex-ParametersHeightforSquare[2])*(ParametersHeightforSquare[1]/ParametersHeightforSquare[0]))+ParametersHeightforSquare[3])/2.0f);


        int limityy = min(yIndex + HalfWindowHeight, height - 1);
        int limitxx = min(xIndex + HalfWindowWidth, width - 1);

        int yy = max(yIndex - HalfWindowHeight, 0);
        int step = (limityy - yy)/4;
        int limitStep = yy + step;

        float H[4];
        float L[4];
        float S[4];
        float n[4];

        bool validWindow = true;

        int totalValidPixels = 0;


        for (int i = 0; i < 4; i++, limitStep+=step){
            H[i] = 0;
            L[i] = 0;
            S[i] = 0;
            n[i] = 0;
            for (; yy < limitStep; yy++){
                for (int xx = max(xIndex - HalfWindowWidth, 0); xx < limitxx; xx++){

                    const int colorIdPixel = yy * colorWidthStep + (3 * xx);
                    const unsigned char h_pixel	= FieldImage[colorIdPixel];
                    const unsigned char l_pixel = FieldImage[colorIdPixel + 1];
                    const unsigned char s_pixel	= FieldImage[colorIdPixel + 2];

                    if (!(h_pixel == 0 && l_pixel == 0 && s_pixel == 0)){
                        n[i]++;
                        H[i] += h_pixel;
                        L[i] += l_pixel;
                        S[i] += s_pixel;
                    }
                }
            }

            H[i] /= n[i];
            L[i] /= n[i];
            S[i] /= n[i];

            totalValidPixels += n[i];

        }

        //        if(n[0] < (2 * HalfWindowWidth * step) * 0.1 ){
        //            validWindow = false;
        //        }
        for (int i = 1; i < 4; ++i) {
            if(n[i] < (2 * HalfWindowWidth * step) * 0.2 ){
                validWindow = false;
                break;
            }
        }


        if(validWindow){
            float percent = totalValidPixels * 100 / (4 * HalfWindowHeight * HalfWindowWidth) * 100000;


            int maxIndex;
            float maxProb = 0;

            for (unsigned k = 0; k < gaussiansNumber; k++){
                int gausPos = GAUSSIAN_LENGTH_W * k + 2;
                float prob = 1;

                for (int i = 1; i < 4; ++i) {
                    int kernelPos = gausPos + 6 * i;

                    float PH = exp( (H[i] - gaussians[kernelPos])*(H[i]-gaussians[kernelPos]) / (-2*gaussians[kernelPos + 1])) / sqrt(2* M_PI *gaussians[kernelPos + 1]);
                    float PL = exp( (L[i] - gaussians[kernelPos + 2])*(L[i]-gaussians[kernelPos + 2]) / (-2*gaussians[kernelPos + 3])) / sqrt(2* M_PI *gaussians[kernelPos + 3]);
                    float PS = exp( (S[i] - gaussians[kernelPos + 4])*(S[i]-gaussians[kernelPos + 4]) / (-2*gaussians[kernelPos + 5])) / sqrt(2* M_PI *gaussians[kernelPos + 5]);
                    prob += PH * PL * PS;
                }

                if(prob == 1)
                    printf("uno\n");

                if(prob > maxProb){
                    maxProb = prob;
                    maxIndex = k;
                }
            }


            if(maxProb > gaussians[GAUSSIAN_LENGTH_W * maxIndex + 1])
            {
                const int gray_tid  = yIndex * width + xIndex;
                PixelClass[gray_tid] = static_cast<unsigned char>(gaussians[GAUSSIAN_LENGTH_W * maxIndex]);
                Probability[gray_tid] = static_cast<float>(maxProb*percent);
            }
        }
    }
}

__global__ void DoubleParticleFilterWindowsBayes_kernel( unsigned char* FieldImage,

                                                         unsigned char* PixelClass,
                                                         float *Probability,
                                                         unsigned char* PixelClass2,
                                                         float *Probability2,

                                                         float *ParametersHeightforSquare,
                                                         float *ParametersWidthforSquare,
                                                         int width,
                                                         int height,
                                                         int colorWidthStep,

                                                         float *gaussians, // Equipo 1, equipo 2 y pasto
                                                         int gaussiansNumber,
                                                         float *gaussians2, // Para porteros y arbitros
                                                         int gaussiansNumber2

                                                         )
{

    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int color_tid = yIndex * colorWidthStep + (3 * xIndex);


    if(xIndex < width && yIndex < height && //xIndex % 4 == 0 && yIndex % 4 == 0 &&
            !(FieldImage[color_tid] == 0 && FieldImage[color_tid + 1] == 0 && FieldImage[color_tid + 2] == 0)){


        const int HalfWindowWidth = (int)round(((((float)yIndex-ParametersWidthforSquare[2])*(ParametersWidthforSquare[1]/ParametersWidthforSquare[0]))+ParametersWidthforSquare[3])/2.0f);
        const int HalfWindowHeight = (int)round(((((float)yIndex-ParametersHeightforSquare[2])*(ParametersHeightforSquare[1]/ParametersHeightforSquare[0]))+ParametersHeightforSquare[3])/2.0f);


        int limityy = min(yIndex + HalfWindowHeight, height - 1);
        int limitxx = min(xIndex + HalfWindowWidth, width - 1);

        int yy = max(yIndex - HalfWindowHeight, 0);
        int step = (limityy - yy)/4;
        int limitStep = yy + step;

        float H[4];
        float L[4];
        float S[4];
        float n[4];
        float e[4];


        bool validWindow = true;

        int totalValidPixels = 0;
        int totalEdgePixels=0;


        for (int i = 0; i < 4; i++, limitStep+=step){
            H[i] = 0;
            L[i] = 0;
            S[i] = 0;
            n[i] = 0;
            e[i] = 0;
            for (; yy < limitStep; yy++){
                for (int xx = max(xIndex - HalfWindowWidth, 0); xx < limitxx; xx++){

                    const int edgepixelid = yy * width + xx;
                    const unsigned char edge_pixel=PixelClass2[edgepixelid];
                    if(edge_pixel>0){
                        e[i]++;
                        totalEdgePixels++;
                    }
                    const int colorIdPixel = yy * colorWidthStep + (3 * xx);
                    const unsigned char h_pixel	= FieldImage[colorIdPixel];
                    const unsigned char l_pixel = FieldImage[colorIdPixel + 1];
                    const unsigned char s_pixel	= FieldImage[colorIdPixel + 2];

                    if (!(h_pixel == 0 && l_pixel == 0 && s_pixel == 0)){
                        n[i]++;
                        H[i] += h_pixel;
                        L[i] += l_pixel;
                        S[i] += s_pixel;
                    }
                }
            }

            H[i] /= n[i];
            L[i] /= n[i];
            S[i] /= n[i];

            totalValidPixels += n[i];
        }

        for (int i = 1; i < 4; ++i) {
            if(n[i] < (2 * HalfWindowWidth * step) * 0.2 ){
                validWindow = false;
                break;
            }
        }

        for (int i = 1; i < 4; ++i) {
            if(e[i] <  1 ){
                validWindow = false;
                break;
            }
        }

        float percentEdge = totalEdgePixels * 100 / (WINDOWS_NUMBER * HalfWindowHeight * HalfWindowWidth);
        if(percentEdge<5){
            validWindow=false;
        }


        if(validWindow){


            float percent = totalValidPixels * 100 / (WINDOWS_NUMBER * HalfWindowHeight * HalfWindowWidth);

            if (percent>40 && percent<70)
                    percent=80;
            else if(percent>=70)
                    percent=100-percent;

           // printf("p: %f\n",percent);

            int maxIndex = 0;
            float maxProb = 0;
            float minDist;



#if 0
            if (xIndex == 2251 && yIndex == 582) {
                printf("--- Point\n");
                printf("x = %d  y = %d\n", xIndex, yIndex);
                printf("--- Window dimensions\n");
                printf("lx = %d  ly = %d\n", HalfWindowWidth*2, HalfWindowHeight*2);
                printf("--- Window\n");
                printf("X: start = %d end = %d\n", max(xIndex - HalfWindowWidth, 0), limitxx);
                printf("Y: start = %d end = %d\n", max(yIndex - HalfWindowHeight, 0), max(yIndex - HalfWindowHeight, 0) + 4*step);
                printf("--- Gaussians\n");
                printf("Number of gaussians = %d\n", gaussiansNumber);
                printf("Gaussian length = %d\n", GAUSSIAN_LENGTH);
                for (unsigned k = 0; k < gaussiansNumber; k++){
                    int gausPos = (GAUSSIAN_LENGTH_W * k) + 2;
                    for (int i = 1; i < 4; ++i) {
                        int kernelPos = gausPos + (6 * i);
                        for (unsigned int p = 0; p < 6; p+=2)
                            printf("%f %f   ", gaussians[kernelPos+p], gaussians[kernelPos+1+p]);
                    }
                }
                printf("--- Color\n");
                printf("Color width step = %d\n", colorWidthStep);
                printf("\n");

                printf("--- Colors in window\n");
                yy = max(yIndex - HalfWindowHeight, 0);
                limitStep = yy + step;
                for (int i = 0; i < 4; i++, limitStep+=step) {
                    for (; yy < limitStep; yy++){
                        for (int xx = max(xIndex - HalfWindowWidth, 0); xx < limitxx; xx++) {
                            const int colorIdPixel = yy * colorWidthStep + (3 * xx);
                            const unsigned char h_pixel	= FieldImage[colorIdPixel];
                            const unsigned char l_pixel = FieldImage[colorIdPixel + 1];
                            const unsigned char s_pixel	= FieldImage[colorIdPixel + 2];
                            if (!(h_pixel == 0 && l_pixel == 0 && s_pixel == 0)) {
                                printf("h = %u   l = %u   s = %u\n", h_pixel, l_pixel, s_pixel);
                            }
                        }
                    }
                }
            }
#endif





            for (unsigned k = 0; k < gaussiansNumber; k++){
                int gausPos = (GAUSSIAN_LENGTH_W * k) + 2;
                float prob = 0;

                for (int i = 1; i < 4; ++i) {
                    int kernelPos = gausPos + (6 * i);

                    /*
                    float PH = exp((-0.5 * (H[i] - gaussians[kernelPos    ])*(H[i]-gaussians[kernelPos    ])) / (gaussians[kernelPos + 1])) / (gaussians[kernelPos + 1]*2.506628);
                    float PL = exp((-0.5 * (L[i] - gaussians[kernelPos + 2])*(L[i]-gaussians[kernelPos + 2])) / (gaussians[kernelPos + 3])) / (gaussians[kernelPos + 3]*2.506628);
                    float PS = exp((-0.5 * (S[i] - gaussians[kernelPos + 4])*(S[i]-gaussians[kernelPos + 4])) / (gaussians[kernelPos + 5])) / (gaussians[kernelPos + 5]*2.506628);
                    //*/

                    // Distances to the mean
                    float PH = (H[i] - gaussians[kernelPos    ])*(H[i] - gaussians[kernelPos    ]);
                    float PL = (L[i] - gaussians[kernelPos + 2])*(L[i] - gaussians[kernelPos + 2]);
                    float PS = (S[i] - gaussians[kernelPos + 4])*(S[i] - gaussians[kernelPos + 4]);
                    //prob  +=  log(1+PH) + log(1+PL) + log(1+PS);
                    prob += PH + PL + PS;

                    //printf("p: %f, ", prob);
                }
                prob=sqrt(prob);

//                if(prob == 1)
//                    printf("uno\n");

//                if(prob > maxProb){
//                    maxProb = prob;
//                    maxIndex = k;
//                    // printf("aquillegamos");
//                }

                if(k==0){
                    minDist=prob;
                    maxIndex = k;
                }
                else if(prob<minDist){
                    minDist=prob;
                    maxIndex = k;
                }
            }

       //     if(maxProb > gaussians[GAUSSIAN_LENGTH_W * maxIndex + 1])
      //      {
                const int gray_tid  = yIndex * width + xIndex;
                PixelClass[gray_tid] = static_cast<unsigned char>(gaussians[GAUSSIAN_LENGTH_W * maxIndex]);
                Probability[gray_tid] = static_cast<float>(minDist);//maxProb);
       //     }

            return;

            maxIndex = maxProb = 0;

            for (unsigned k = 0; k < gaussiansNumber2; k++){
                int gausPos = GAUSSIAN_LENGTH_W * k + 2;
                float prob = 0;

                for (int i = 1; i < 4; ++i) {
                    int kernelPos = gausPos + 6 * i;

                    float PH = exp( (H[i] - gaussians2[kernelPos])*(H[i]-gaussians2[kernelPos]) / (-2*gaussians2[kernelPos + 1])) / sqrt(2* M_PI *gaussians2[kernelPos + 1]);
                    float PL = exp( (L[i] - gaussians2[kernelPos + 2])*(L[i]-gaussians2[kernelPos + 2]) / (-2*gaussians2[kernelPos + 3])) / sqrt(2* M_PI *gaussians2[kernelPos + 3]);
                    float PS = exp( (S[i] - gaussians2[kernelPos + 4])*(S[i]-gaussians2[kernelPos + 4]) / (-2*gaussians2[kernelPos + 5])) / sqrt(2* M_PI *gaussians2[kernelPos + 5]);
                    prob += PH * PL * PS;
                }

                if(prob == 1)
                    printf("uno\n");

                if(prob > maxProb){
                    maxProb = prob;
                    maxIndex = k;
                }
            }

            if(maxProb > gaussians2[GAUSSIAN_LENGTH_W * maxIndex + 1])
            {
                const int gray_tid  = yIndex * width + xIndex;
                PixelClass2[gray_tid] = static_cast<unsigned char>(gaussians2[GAUSSIAN_LENGTH_W * maxIndex]);
                Probability2[gray_tid] = static_cast<float>(maxProb*percent);
            }


        }
    }
}


__global__ void DoubleParticleFilterWindowsHistogram_kernel( unsigned char* FieldImage,

                                                             unsigned char* PixelClass,
                                                             float *Probability,

                                                             unsigned char* PixelClass2,
                                                             float *Probability2,

                                                             int numberModels,
                                                             float* histograms,

                                                             int numberModels2,
                                                             float* histograms2,

                                                             float *ParametersHeightforSquare,
                                                             float *ParametersWidthforSquare,
                                                             int width,
                                                             int height,
                                                             int colorWidthStep


                                                             //                                                             unsigned char* PixelClass2,
                                                             //                                                             float *Probability2,

                                                             //                                                             float *gaussians,
                                                             //                                                             int gaussiansNumber,

                                                             //                                                             float *gaussians2,
                                                             //                                                             int gaussiansNumber2

                                                             )
{


    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

    if(xIndex < width && yIndex < height && //xIndex % 4 == 0 && yIndex % 4 == 0 &&
            !(FieldImage[color_tid] == 0 && FieldImage[color_tid + 1] == 0 && FieldImage[color_tid + 2] == 0)){


        const int gray_tid  = yIndex * width + xIndex;

        const int prob_tid  = gray_tid; //yIndex * width + xIndex;

        const int HalfWindowWidth = (int)round(((((float)yIndex-ParametersWidthforSquare[2])*(ParametersWidthforSquare[1]/ParametersWidthforSquare[0]))+ParametersWidthforSquare[3])/2.0f);
        const int HalfWindowHeight = (int)round(((((float)yIndex-ParametersHeightforSquare[2])*(ParametersHeightforSquare[1]/ParametersHeightforSquare[0]))+ParametersHeightforSquare[3])/2.0f);

        int limityy = min(yIndex + HalfWindowHeight, height - 1);
        int limitxx = min(xIndex + HalfWindowWidth, width - 1);

        int yy = max(yIndex - HalfWindowHeight, 0);
        int step = (limityy - yy)/WINDOWS_NUMBER;
        int limitStep = yy + step;

        int validNumber[WINDOWS_NUMBER];

        int histogramH[WINDOWS_NUMBER][NBINS];
        int histogramL[WINDOWS_NUMBER][NBINS];
        int histogramS[WINDOWS_NUMBER][NBINS];

        bool validWindow = true;

        int totalValidPixels = 0;

        const int BYTES_HIST = NBINS * sizeof(int);


        ///Calcula histograma de esta particula
        for (int i = 0; i < WINDOWS_NUMBER; i++, limitStep+=step){

            memset(histogramH[i], 0, BYTES_HIST);
            memset(histogramL[i], 0, BYTES_HIST);
            memset(histogramS[i], 0, BYTES_HIST);

            validNumber[i] = 0;

            for (; yy < limitStep; yy++){
                for (int xx = max(xIndex - HalfWindowWidth, 0); xx < limitxx; xx++){

                    const int colorIdPixel = yy * colorWidthStep + (3 * xx);
                    const unsigned char h_pixel	= FieldImage[colorIdPixel];
                    const unsigned char l_pixel = FieldImage[colorIdPixel + 1];
                    const unsigned char s_pixel	= FieldImage[colorIdPixel + 2];

                    if (!(h_pixel == 0 && l_pixel == 0 && s_pixel == 0))
                    {
                        validNumber[i]++;
                        histogramH[i][(int)(h_pixel * NBINS/RANGE_H)]++;
                        histogramL[i][(int)(l_pixel * NBINS/RANGE_L)]++;
                        histogramS[i][(int)(s_pixel * NBINS/RANGE_S)]++;
                    }
                }
            }
            totalValidPixels += validNumber[i];
        }


        /// Validar ventana por cantidad de pixeles, puede morir

//        if(validNumber[0] < (2 * HalfWindowWidth * step) * 0.1 ){
//            validWindow = false;
//        }
        for (int i = 1; i < WINDOWS_NUMBER; ++i) {
            if(validNumber[i] < (2 * HalfWindowWidth * step) * 0.20 ){
                validWindow = false;
                break;
            }
        }


        if(validWindow){
            float percent = totalValidPixels * 100 / (WINDOWS_NUMBER * HalfWindowHeight * HalfWindowWidth);
            if (percent>40 && percent<70)
                    percent=80;
            else if(percent>=70)
                    percent=100-percent;

           // printf("p: %f\n",percent);

            float* distances = new float[numberModels];

            ///Checar si se parecen los histogramas
            for(int i = 0; i < numberModels; i++){
                distances[i] = 0;
                for (int k = 1; k < WINDOWS_NUMBER; k++){
                    int histogramPosition = (FEATURES_SIZE * i) + (k * HIST_SIZE);
                    for (int j = 0; j < NBINS; j++){
                        distances[i] += sqrtf((histogramH[k][j]/(float)validNumber[k]) * histograms[histogramPosition + j]);
                        distances[i] += sqrtf((histogramL[k][j]/(float)validNumber[k]) * histograms[histogramPosition + j + (NBINS)]);
                        distances[i] += sqrtf((histogramS[k][j]/(float)validNumber[k]) * histograms[histogramPosition + j + (2 * NBINS)]);
                    }
                }

                //  distances[n] = distances[n]/((float)NBINS*6.0);

                distances[i] = (1-(distances[i]/(3*(WINDOWS_NUMBER-1)))); //* SumaValidPixels;
               //   distances[i] = (3*(WINDOWS_NUMBER-1))-(distances[i]); //* SumaValidPixels;
            }


            float minDistance = distances[0];
            int minIndex = 0;
            for(int i = 1; i < numberModels; i++){
                if(distances[i] < minDistance){
                    minDistance = distances[i];
                    minIndex = i;
                }
            }



            PixelClass[gray_tid] = static_cast<unsigned char>(histograms[FEATURES_SIZE * minIndex]);
            Probability[prob_tid] = static_cast<float>((minDistance * percent)); //+sqrtf(yIndex/1000)


            delete[] distances;


            return;
            if(xIndex > (width/3) && xIndex < 2*(width/3))
                    return;
                    //&& yIndex < height && //xIndex % 4 == 0 && yIndex % 4 == 0 &&



            distances = new float[numberModels];
            ///Checar si se parecen los histogramas (Porteros y arbitros)
            for(int i = 0; i < numberModels2; i++){
                distances[i] = 0;
                for (int k = 1; k < WINDOWS_NUMBER; k++){
                    int histogramPosition = FEATURES_SIZE * i + k * HIST_SIZE;
                    for (int j = 0; j < NBINS; j++){
                        distances[i] += sqrtf((histogramH[k][j]/(float)totalValidPixels) * histograms2[histogramPosition + j]);
                        distances[i] += sqrtf((histogramL[k][j]/(float)totalValidPixels) * histograms2[histogramPosition + j + (NBINS)]);
                        distances[i] += sqrtf((histogramS[k][j]/(float)totalValidPixels) * histograms2[histogramPosition + j + (2 * NBINS)]);
                    }
                }

                //  distances[n] = distances[n]/((float)NBINS*6.0);

                distances[i] = (1-(distances[i]/(3*(WINDOWS_NUMBER-1)))); //* SumaValidPixels;
            }



            minDistance = distances[0];
            minIndex = 0;
            for(int i = 1; i < numberModels2; i++){
                if(distances[i] < minDistance){
                    minDistance = distances[i];
                    minIndex = i;
                }
            }

            PixelClass2[gray_tid] = static_cast<unsigned char>(histograms2[FEATURES_SIZE * minIndex]);
            Probability2[prob_tid] = static_cast<float>(minDistance * percent);

            delete[] distances;

        }
    }
}


void ReserveCudaMemoryBayes(std::vector<std::vector<float> > gaussians, std::vector<float> ParametersHeightForSquare, std::vector<float> ParametersWidthForSquare, cv::Mat FieldImage, cv::Mat PixelClass, cv::Mat Probability){

    int ParametersForSquareBytes = 4 * sizeof(float);
    int gaussiansSize = gaussians.size() * GAUSSIAN_LENGTH_W * sizeof(float);

    numberOfGaussians = gaussians.size();

    colorBytes = FieldImage.step * FieldImage.rows;
    grayBytes = PixelClass.step * PixelClass.rows;
    ProbBytes = Probability.cols * Probability.rows * sizeof(float) ;

    float *h_gaussians = new float[numberOfGaussians * GAUSSIAN_LENGTH_W];

    for(int i = 0; i < numberOfGaussians; i++){
        for(int j = 0; j < GAUSSIAN_LENGTH_W; j++){
            h_gaussians[j + i * GAUSSIAN_LENGTH_W] = gaussians[i][j];
        }
    }


    SAFE_CALL(cudaMalloc<float>(&d_gaussians, gaussiansSize),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_gaussians, h_gaussians, gaussiansSize, cudaMemcpyHostToDevice),"CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc<float>(&d_ParametersHeightForSquare, ParametersForSquareBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_ParametersWidthForSquare, ParametersForSquareBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_ParametersHeightForSquare, ParametersHeightForSquare.data(), ParametersForSquareBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_ParametersWidthForSquare, ParametersWidthForSquare.data(), ParametersForSquareBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    SAFE_CALL(cudaMalloc<unsigned char>(&d_FieldImage, colorBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&d_PixelClass, grayBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_Probability, ProbBytes),"CUDA Malloc Failed");

    delete[] h_gaussians;
}


void ReserveCudaMemoryBayes(std::vector<std::vector<float> > gaussians,std::vector<std::vector<float> > gaussians2, std::vector<float> ParametersHeightForSquare, std::vector<float> ParametersWidthForSquare, cv::Mat FieldImage, cv::Mat PixelClass, cv::Mat Probability){

    int ParametersForSquareBytes = 4 * sizeof(float);

    int gaussiansSize = gaussians.size() * GAUSSIAN_LENGTH_W * sizeof(float);
    int gaussiansSize2 = gaussians2.size() * GAUSSIAN_LENGTH_W * sizeof(float);

    numberOfGaussians = gaussians.size();
    numberOfGaussians2 = gaussians2.size();

    colorBytes = FieldImage.step * FieldImage.rows;
    grayBytes = PixelClass.step * PixelClass.rows;
    ProbBytes = Probability.cols * Probability.rows * sizeof(float) ;

    float *h_gaussians = new float[numberOfGaussians * GAUSSIAN_LENGTH_W];

    for(int i = 0; i < numberOfGaussians; i++){
        for(int j = 0; j < GAUSSIAN_LENGTH_W; j++){
            h_gaussians[j + i * GAUSSIAN_LENGTH_W] = gaussians[i][j];
        }
    }

    float *h_gaussians2 = new float[numberOfGaussians2 * GAUSSIAN_LENGTH_W];

    for(int i = 0; i < numberOfGaussians2; i++){
        for(int j = 0; j < GAUSSIAN_LENGTH_W; j++){
            h_gaussians2[j + i * GAUSSIAN_LENGTH_W] = gaussians2[i][j];
        }
    }


    SAFE_CALL(cudaMalloc<float>(&d_gaussians, gaussiansSize),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_gaussians, h_gaussians, gaussiansSize, cudaMemcpyHostToDevice),"CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc<float>(&d_gaussians2, gaussiansSize2),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_gaussians2, h_gaussians2, gaussiansSize2, cudaMemcpyHostToDevice),"CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc<float>(&d_ParametersHeightForSquare, ParametersForSquareBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_ParametersWidthForSquare, ParametersForSquareBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(d_ParametersHeightForSquare, ParametersHeightForSquare.data(), ParametersForSquareBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_ParametersWidthForSquare, ParametersWidthForSquare.data(), ParametersForSquareBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    SAFE_CALL(cudaMalloc<unsigned char>(&d_FieldImage, colorBytes),"CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc<unsigned char>(&d_PixelClass, grayBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_Probability, ProbBytes),"CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc<unsigned char>(&d_PixelClass2, grayBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<float>(&d_Probability2, ProbBytes),"CUDA Malloc Failed");

    delete[] h_gaussians;
    delete[] h_gaussians2;
}

void FreeCudaMemoryBayes(){

    //Free the device memory
    SAFE_CALL(cudaFree(d_gaussians),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_ParametersHeightForSquare),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_ParametersWidthForSquare),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_FieldImage),"CUDA Malloc Failed");
    SAFE_CALL(cudaFree(d_PixelClass),"CUDA Malloc Failed");
    SAFE_CALL(cudaFree(d_Probability),"CUDA Malloc Failed");
}

__global__ void ParticleFilterPixelsBayes_kernel( unsigned char* FieldImage,
                                                  unsigned char* PixelClass,
                                                  float *Probability,
                                                  int width,
                                                  int height,
                                                  int colorWidthStep,
                                                  float *gaussians,
                                                  int gaussiansNumber
                                                  )
{

    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

    if(xIndex < width && yIndex < height &&
            !(FieldImage[color_tid] == 0 && FieldImage[color_tid + 1] == 0 && FieldImage[color_tid + 2] == 0)){

        const int colorIdPixel = yIndex * colorWidthStep + (3 * xIndex);
        const unsigned char H = FieldImage[colorIdPixel];
        const unsigned char L = FieldImage[colorIdPixel + 1];
        const unsigned char S = FieldImage[colorIdPixel + 2];

        if(!(H==0 && L==0 && S==0)){

            int maxIndex;
            double maxProb = 0;

            for (int k = 0; k < gaussiansNumber; k++){
                int gausPos = GAUSSIAN_LENGTH * k;

                double PH = exp( (H - gaussians[gausPos + 2])*(H-gaussians[gausPos + 2]) / (-2*gaussians[gausPos + 3])) / sqrt(2* M_PI *gaussians[gausPos + 3]);
                double PL = exp( (L - gaussians[gausPos + 4])*(L-gaussians[gausPos + 4]) / (-2*gaussians[gausPos + 5])) / sqrt(2* M_PI *gaussians[gausPos + 5]);
                double PS = exp( (S - gaussians[gausPos + 6])*(S-gaussians[gausPos + 6]) / (-2*gaussians[gausPos + 7])) / sqrt(2* M_PI *gaussians[gausPos + 7]);
                double prob =  PH * PL * PS;
                //printf("%f %f %f = %f\n",PH,PL,PS,prob);

                if(gaussians[gausPos] == 1 && prob > 0){
                    maxProb = prob;
                    maxIndex = k;
                    //printf("p: %f\n",prob);
                    break;

                }

                if(prob > maxProb){
                    maxProb = prob;
                    maxIndex = k;
                }
            }

            //printf("prob: %f k: %d\n", maxProb, maxIndex); //gaussians[GAUSSIAN_LENGTH * maxIndex]);

            //printf("%f\n",gaussians[GAUSSIAN_LENGTH * maxIndex]);

            const int gray_tid  = yIndex * width + xIndex;
            PixelClass[gray_tid] = static_cast<unsigned char>(gaussians[GAUSSIAN_LENGTH * maxIndex]);
            Probability[gray_tid] = static_cast<float>(maxProb);
        }

    }
}

__global__ void RemoveByClass_kernel( unsigned char* FieldImage,
                                      int width,
                                      int height,
                                      int colorWidthStep,
                                      float *gaussians,
                                      int gaussiansNumber
                                      )
{

    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

    if(xIndex < width && yIndex < height && //xIndex % 4 == 0 && yIndex % 4 == 0 &&
            !(FieldImage[color_tid] == 0 && FieldImage[color_tid + 1] == 0 && FieldImage[color_tid + 2] == 0)){


        const unsigned char H = FieldImage[color_tid];
        const unsigned char L = FieldImage[color_tid + 1];
        const unsigned char S = FieldImage[color_tid + 2];

        for (int k = 0; k < gaussiansNumber; k++){


            int gausPos = GAUSSIAN_LENGTH_W * k;
            double PH = exp( (H - gaussians[gausPos + 2])*(H-gaussians[gausPos + 2]) / (-2*gaussians[gausPos + 3])) / sqrt(2* M_PI *gaussians[gausPos + 3]);
            double PL = exp( (L - gaussians[gausPos + 4])*(L-gaussians[gausPos + 4]) / (-2*gaussians[gausPos + 5])) / sqrt(2* M_PI *gaussians[gausPos + 5]);
            double PS = exp( (S - gaussians[gausPos + 6])*(S-gaussians[gausPos + 6]) / (-2*gaussians[gausPos + 7])) / sqrt(2* M_PI *gaussians[gausPos + 7]);
            double prob =  PH * PL * PS;

            if(gaussians[gausPos] == 1 && prob > 0){


                FieldImage[color_tid] = 0;
                FieldImage[color_tid + 1] = 0;
                FieldImage[color_tid + 2] = 0;
                break;
            }
        }
    }
}






__global__ void nonMaximumSuppression_kernel(
        unsigned char* PixelClass,
        float *Probability,
        float *ParametersHeightforSquare,
        float *ParametersWidthforSquare,
        int width,
        int height
        )
{

    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int gray_tid  = yIndex * width + xIndex;

    if(xIndex < width && yIndex < height && PixelClass[gray_tid] != 0 //&& xIndex % 4 == 0 && yIndex % 4 == 0
            ){

        const int NeighborhoodX = (int)((((float)yIndex-ParametersWidthforSquare[2])*(ParametersWidthforSquare[1]/ParametersWidthforSquare[0]))+ParametersWidthforSquare[3]);
        const int NeighborhoodY = (int)((((float)yIndex-ParametersHeightforSquare[2])*(ParametersHeightforSquare[1]/ParametersHeightforSquare[0]))+ParametersHeightforSquare[3]);

        int ClassValue = PixelClass[gray_tid];
        double ProbValue = Probability[gray_tid];
        bool IsMaximum = true;
        for (int yy = yIndex - NeighborhoodY; IsMaximum && yy <= yIndex + NeighborhoodY; yy++) {
            for (int xx= xIndex - NeighborhoodX; IsMaximum && xx <= xIndex + NeighborhoodX; xx++){

                if (!(yy == yIndex && xx == xIndex) && yy < height && xx < width && yy > 0 && xx > 0){

                    const int pixelPosition = yy * width + xx;

                //   if (PixelClass[pixelPosition]>0 && Probability[pixelPosition] >ProbValue){
                    if (PixelClass[pixelPosition]>0 && Probability[pixelPosition] < ProbValue) {
                  //  if ((PixelClass[pixelPosition] == ClassValue && Probability[pixelPosition] < ProbValue)){
                        PixelClass[gray_tid] = 0;
                        IsMaximum = false;
                        return;
                    }
                }
            }
        }
    }
}




__global__ void RemoveByClassWindow_kernel( unsigned char* FieldImage,
                                            unsigned char* PixelClass,
                                            float *Probability,
                                            float *ParametersHeightforSquare,
                                            float *ParametersWidthforSquare,
                                            int width,
                                            int height,
                                            int colorWidthStep,
                                            float *gaussians,
                                            int gaussiansNumber
                                            )
{



    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

    if(xIndex < width && yIndex < height && //xIndex % 4 == 0 && yIndex % 4 == 0 &&
            !(FieldImage[color_tid] == 0 && FieldImage[color_tid + 1] == 0 && FieldImage[color_tid + 2] == 0)){


        const int HalfWindowWidth =1;//(int)round(((((float)yIndex-ParametersWidthforSquare[2])*(ParametersWidthforSquare[1]/ParametersWidthforSquare[0]))+ParametersWidthforSquare[3])/2.0f);
        const int HalfWindowHeight = 1;//(int)round(((((float)yIndex-ParametersHeightforSquare[2])*(ParametersHeightforSquare[1]/ParametersHeightforSquare[0]))+ParametersHeightforSquare[3])/2.0f);

        float H = 0;
        float L = 0;
        float S = 0;
        float n = 0;
        int limityy = min(yIndex + HalfWindowHeight, height - 1);
        int limitxx = min(xIndex + HalfWindowWidth, width - 1);

        for (int yy = max(yIndex - HalfWindowHeight, 0); yy < limityy; yy++){
            for (int xx = max(xIndex - HalfWindowWidth, 0); xx < limitxx; xx++){

                const int colorIdPixel = yy * colorWidthStep + (3 * xx);
                const unsigned char h_pixel	= FieldImage[colorIdPixel];
                const unsigned char l_pixel = FieldImage[colorIdPixel + 1];
                const unsigned char s_pixel	= FieldImage[colorIdPixel + 2];

                if (!(h_pixel == 0 && l_pixel == 0 && s_pixel == 0)){
                    n++;
                    H += h_pixel;
                    L += l_pixel;
                    S += s_pixel;
                }
            }
        }



        H /= n;
        L /= n;
        S /= n;

        //        FieldImage[color_tid] = 0;
        //        FieldImage[color_tid + 1] = 0;
        //        FieldImage[color_tid + 2] = 0;



        for (unsigned k = 0; k < gaussiansNumber; k++){
            int gausPos = GAUSSIAN_LENGTH * k;

            float PH = exp( (H - gaussians[gausPos + 2])*(H-gaussians[gausPos + 2]) / (-2*gaussians[gausPos + 3])) / sqrt(2* M_PI *gaussians[gausPos + 3]);
            float PL = exp( (L - gaussians[gausPos + 4])*(L-gaussians[gausPos + 4]) / (-2*gaussians[gausPos + 5])) / sqrt(2* M_PI *gaussians[gausPos + 5]);
            float PS = exp( (S - gaussians[gausPos + 6])*(S-gaussians[gausPos + 6]) / (-2*gaussians[gausPos + 7])) / sqrt(2* M_PI *gaussians[gausPos + 7]);
            float prob = PH * PL * PS;

            if(gaussians[gausPos] == 1 && prob > 0){
                //                FieldImage[color_tid] = 180;
                //                FieldImage[color_tid + 1] = 255;
                //                FieldImage[color_tid + 2] = 255;
                FieldImage[color_tid] = 0;
                FieldImage[color_tid + 1] = 0;
                FieldImage[color_tid + 2] = 0;
                break;
            }

        }

    }





}


void ParticleFilterBayesCUDA(cv::Mat &FieldImage, cv::Mat& PixelClass, cv::Mat& Probability)
{

    SAFE_CALL(cudaMemset(d_PixelClass, 0, grayBytes),"CUDA Memset Failed");
    SAFE_CALL(cudaMemcpy(d_FieldImage, FieldImage.ptr(), colorBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    //Specify a reasonable block size
    const dim3 block(BLOCKSIZE ,BLOCKSIZE);
    //Calculate grid size to cover the whole image
    const dim3 grid((FieldImage.cols + block.x - 1)/block.x, (FieldImage.rows + block.y - 1)/block.y);

    RemoveByClass_kernel<<<grid, block>>>(d_FieldImage, FieldImage.cols, FieldImage.rows, FieldImage.step, d_gaussians, numberOfGaussians);

    //    RemoveByClassWindow_kernel<<<grid, block>>>(d_FieldImage, d_PixelClass, d_Probability, d_ParametersHeightForSquare, d_ParametersWidthForSquare,
    //                                                FieldImage.cols, FieldImage.rows, FieldImage.step, d_gaussians, numberOfGaussians);

    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    //    cv::Mat img = FieldImage.clone();
    //    SAFE_CALL(cudaMemcpy(img.ptr(), d_FieldImage, colorBytes, cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
    //    cv::cvtColor(img, img, CV_HLS2BGR);
    //    cv::imshow("img", img);

    ParticleFilterWindowsBayes_kernel<<<grid, block>>>(d_FieldImage, d_PixelClass, d_Probability, d_ParametersHeightForSquare, d_ParametersWidthForSquare,
                                                       FieldImage.cols, FieldImage.rows, FieldImage.step, d_gaussians, numberOfGaussians);

    //    ParticleFilterPixelsBayes_kernel<<<grid, block>>>(d_FieldImage, d_PixelClass, d_Probability,
    //                                                      FieldImage.cols, FieldImage.rows, FieldImage.step, d_gaussians, numberOfGaussians);

    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    nonMaximumSuppression_kernel<<<grid, block>>>(d_PixelClass, d_Probability, d_ParametersHeightForSquare, d_ParametersWidthForSquare,FieldImage.cols, FieldImage.rows);

    //Synchronize to check for any kernel launch errors
    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    SAFE_CALL(cudaMemcpy(PixelClass.ptr(), d_PixelClass,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(Probability.ptr(), d_Probability,ProbBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

}



void ParticleFilterBayesCUDA(cv::Mat& FieldImage, cv::Mat& PixelClass, cv::Mat& Probability, cv::Mat& PixelClass2, cv::Mat& Probability2)
{


    SAFE_CALL(cudaMemcpy(d_FieldImage, FieldImage.ptr(), colorBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    SAFE_CALL(cudaMemset(d_PixelClass, 0, grayBytes),"CUDA Memset Failed");


    SAFE_CALL(cudaMemcpy(d_PixelClass2, PixelClass2.ptr(), grayBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");


    //Specify a reasonable block size
    const dim3 block(BLOCKSIZE ,BLOCKSIZE);
    //Calculate grid size to cover the whole image
    const dim3 grid((FieldImage.cols + block.x - 1)/block.x, (FieldImage.rows + block.y - 1)/block.y);




   RemoveByClass_kernel<<<grid, block>>>(d_FieldImage, FieldImage.cols, FieldImage.rows, FieldImage.step, d_gaussians, numberOfGaussians);

    //    RemoveByClassWindow_kernel<<<grid, block>>>(d_FieldImage, d_PixelClass, d_Probability, d_ParametersHeightForSquare, d_ParametersWidthForSquare,
    //                                                FieldImage.cols, FieldImage.rows, FieldImage.step, d_gaussians, numberOfGaussians);

    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

//    cv::Mat img = FieldImage.clone();
//    SAFE_CALL(cudaMemcpy(img.ptr(), d_FieldImage, colorBytes, cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
//    cv::cvtColor(img, img, CV_HLS2BGR);
//    cv::imshow("img", img);


    DoubleParticleFilterWindowsBayes_kernel<<<grid, block>>>(d_FieldImage, d_PixelClass, d_Probability, d_PixelClass2, d_Probability2, d_ParametersHeightForSquare, d_ParametersWidthForSquare,
                                                             FieldImage.cols, FieldImage.rows, FieldImage.step, d_gaussians, numberOfGaussians, d_gaussians2, numberOfGaussians2);


    //    ParticleFilterPixelsBayes_kernel<<<grid, block>>>(d_FieldImage, d_PixelClass, d_Probability,
    //                                                      FieldImage.cols, FieldImage.rows, FieldImage.step, d_gaussians, numberOfGaussians);

    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    SAFE_CALL(cudaMemset(d_PixelClass2, 0, grayBytes),"CUDA Memset Failed");


    nonMaximumSuppression_kernel<<<grid, block>>>(d_PixelClass, d_Probability, d_ParametersHeightForSquare, d_ParametersWidthForSquare,FieldImage.cols, FieldImage.rows);

    //Synchronize to check for any kernel launch errors
    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    nonMaximumSuppression_kernel<<<grid, block>>>(d_PixelClass2, d_Probability2, d_ParametersHeightForSquare, d_ParametersWidthForSquare,FieldImage.cols, FieldImage.rows);

    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    SAFE_CALL(cudaMemcpy(PixelClass.ptr(), d_PixelClass,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(Probability.ptr(), d_Probability,ProbBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

    SAFE_CALL(cudaMemcpy(PixelClass2.ptr(), d_PixelClass2,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(Probability2.ptr(), d_Probability2,ProbBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

    SAFE_CALL(cudaMemcpy(FieldImage.ptr(), d_FieldImage, colorBytes, cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

}




void ParticleFilterHistCUDA(cv::Mat& FieldImage, cv::Mat& PixelClass, cv::Mat& Probability, cv::Mat& PixelClass2, cv::Mat& Probability2)
{


    SAFE_CALL(cudaMemcpy(d_FieldImage, FieldImage.ptr(), colorBytes, cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

    SAFE_CALL(cudaMemset(d_PixelClass, 0, grayBytes),"CUDA Memset Failed");

    SAFE_CALL(cudaMemset(d_PixelClass2, 0, grayBytes),"CUDA Memset Failed");

    //Specify a reasonable block size
    const dim3 block(BLOCKSIZE ,BLOCKSIZE);
    //Calculate grid size to cover the whole image
    const dim3 grid((FieldImage.cols + block.x - 1)/block.x, (FieldImage.rows + block.y - 1)/block.y);

    RemoveByClass_kernel<<<grid, block>>>(d_FieldImage, FieldImage.cols, FieldImage.rows, FieldImage.step, d_gaussians, numberOfGaussians);

    //    RemoveByClassWindow_kernel<<<grid, block>>>(d_FieldImage, d_PixelClass, d_Probability, d_ParametersHeightForSquare, d_ParametersWidthForSquare,
    //                                                FieldImage.cols, FieldImage.rows, FieldImage.step, d_gaussians, numberOfGaussians);

    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    //cv::Mat img = FieldImage.clone();
    //SAFE_CALL(cudaMemcpy(img.ptr(), d_FieldImage, colorBytes, cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
    //cv::cvtColor(img, img, CV_HLS2BGR);
    //cv::imshow("img", img);


    DoubleParticleFilterWindowsHistogram_kernel<<<grid, block>>>(d_FieldImage, d_PixelClass, d_Probability, d_PixelClass2, d_Probability2,
                                                                 numberHistograms, d_histograms, numberHistograms2, d_histograms2,
                                                                 d_ParametersHeightForSquare, d_ParametersWidthForSquare,
                                                                 FieldImage.cols, FieldImage.rows, FieldImage.step);



    //    ParticleFilterPixelsBayes_kernel<<<grid, block>>>(d_FieldImage, d_PixelClass, d_Probability,
    //                                                      FieldImage.cols, FieldImage.rows, FieldImage.step, d_gaussians, numberOfGaussians);

    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    nonMaximumSuppression_kernel<<<grid, block>>>(d_PixelClass, d_Probability, d_ParametersHeightForSquare, d_ParametersWidthForSquare,FieldImage.cols, FieldImage.rows);

    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

//    nonMaximumSuppression_kernel<<<grid, block>>>(d_PixelClass2, d_Probability2, d_ParametersHeightForSquare, d_ParametersWidthForSquare,FieldImage.cols, FieldImage.rows);

//    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    SAFE_CALL(cudaMemcpy(PixelClass.ptr(), d_PixelClass,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(Probability.ptr(), d_Probability,ProbBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

    SAFE_CALL(cudaMemcpy(PixelClass2.ptr(), d_PixelClass2,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(Probability2.ptr(), d_Probability2,ProbBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

    SAFE_CALL(cudaMemcpy(FieldImage.ptr(), d_FieldImage, colorBytes, cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

}








