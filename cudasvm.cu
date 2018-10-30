#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <common_functions.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "libsvm/svm.h"

// for get_Q()
struct svm_node** d_x = NULL;
double *d_x_square = NULL;
signed char* d_y = NULL;
svm_node* g_x_space = NULL;
__constant__ __device__ struct svm_parameter d_svm_parameter;
__constant__ __device__ struct svm_node* d_x_space;
__constant__ __device__ struct svm_node* host_x_space;

// for CUDA_k_function()
svm_node** d_SV = NULL;
double* d_output;
__constant__ __device__ struct svm_parameter d_model_parameter;

__device__ double dot(const svm_node *px, const svm_node *py){
    double sum = 0;
    while(px->index != -1 && py->index != -1){
        if(px->index == py->index){
            sum += px->value * py->value;
            ++px;
            ++py;
        }else{
            if(px->index > py->index){
                ++py;
            }else{
                ++px;
            }
        }
    }
    return sum;
}

__device__ static double powi(double base, int times){
    double tmp = base, ret = 1.0;
    for(int t=times; t>0; t/=2){
        if(t%2==1){
            ret*=tmp;
        }
        tmp = tmp * tmp;
    }
    return ret;
}

__global__ void get_Q(struct svm_node** CUDA_x,signed char *CUDA_y, double *CUDA_x_square, int x, int starty, int problen, float *output){
	int y = blockDim.x*blockIdx.x + threadIdx.x + starty;
    if( y>=problen ){
		return;
    }
	const svm_node *px = d_x_space + (CUDA_x[x] - host_x_space);
	const svm_node *py = d_x_space + (CUDA_x[y] - host_x_space);
	double value = 0;
	
	switch(d_svm_parameter.kernel_type){
		case LINEAR:
			value = dot(px,py);
			break;
		case POLY:
			value = powi(d_svm_parameter.gamma*dot(px,py)+d_svm_parameter.coef0,d_svm_parameter.degree);
			break;
		case RBF:
			value = exp(-d_svm_parameter.gamma*(CUDA_x_square[x]+CUDA_x_square[y]-2*dot(px,py)));
			break;
		case SIGMOID:
			value = tanh(d_svm_parameter.gamma*dot(px,py)+d_svm_parameter.coef0);
			break;
		default:
			break;
	}
	output[y-starty] = (float)(value*CUDA_y[x]*CUDA_y[y]);
}

static int has_init = 0;

void CUDA_init_model(const struct svm_node* x_space, int problen){
    if( !has_init ){
        findCudaDevice(1, NULL);
		has_init = 1;
		size_t elements = 0;
		const struct svm_node* pNode = x_space;
		for( int i=0; i<problen; i++ ){
			while( pNode->index!=-1 ){
				elements++;
				pNode++;
			}
			pNode++;
			elements++;
		}
		checkCudaErrors(cudaMalloc((void **)&g_x_space, sizeof(struct svm_node)*elements));
        checkCudaErrors(cudaMemcpy(g_x_space, x_space, sizeof(struct svm_node)*elements, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyToSymbol(d_x_space, &g_x_space, sizeof(g_x_space)));
		checkCudaErrors(cudaMemcpyToSymbol(host_x_space, &x_space, sizeof(x_space)));
	}
}

void CUDA_uninit_model(){
	checkCudaErrors(cudaFree(g_x_space));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_SV));
    d_SV = NULL;
    d_output = NULL;
}

void CUDA_init_SVC_Q(int problen, const struct svm_node** x, const signed char* y, double* x_square, const svm_parameter& svm_parameter){
	checkCudaErrors(cudaMalloc((void ***)&d_y, sizeof(signed char)*problen));
	checkCudaErrors(cudaMemcpy(d_y, y, sizeof(signed char)*problen, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void ***)&d_x_square, sizeof(double)*problen));
	checkCudaErrors(cudaMemcpy(d_x_square, x_square, sizeof(double)*problen, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void ***)&d_x, sizeof(struct svm_node*)*problen));
	checkCudaErrors(cudaMemcpy(d_x, x, sizeof(struct svm_node*)*problen, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_svm_parameter, &svm_parameter, sizeof(svm_parameter)));
}

void CUDA_uninit_SVC_Q(){
	checkCudaErrors(cudaFree(d_y));
	checkCudaErrors(cudaFree(d_x_square));
	checkCudaErrors(cudaFree(d_x));
	d_y = NULL;
	d_x_square = NULL;
	d_x = NULL;
}

void CUDA_get_Q(int x, int starty, int endy, float* output){
    const int threadPerBlock = 256;
    const int blockPerGrid = (endy-starty+threadPerBlock-1)/threadPerBlock;
	float* d_output;
	checkCudaErrors(cudaMalloc((void **)&d_output, sizeof(float)*(endy-starty)));
	get_Q<<<blockPerGrid, threadPerBlock>>>(d_x, d_y, d_x_square, x, starty, endy-starty, d_output);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(output+starty, d_output, sizeof(float)*(endy-starty), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_output));
}

__global__ void k_function(const svm_node* x, svm_node** SV, int modellen, double* d_output){
    int y = blockDim.x*blockIdx.x + threadIdx.x;
    if( y>=modellen )
        return;

	svm_node* px = d_x_space + (x-host_x_space);
	const svm_node *py = d_x_space + (SV[y]-host_x_space);

    switch(d_model_parameter.kernel_type){
        case LINEAR:
            d_output[y] = dot(px,py);
            break;
        case POLY:
            d_output[y] = powi(d_model_parameter.gamma*dot(px,py)+d_model_parameter.coef0,d_model_parameter.degree);
            break;
		case SIGMOID:
			d_output[y] = tanh(d_model_parameter.gamma*dot(px,py)+d_model_parameter.coef0);
			break;
		case RBF:
		{
			double sum = 0;
			while(px->index != -1 && py->index !=-1){
				if(px->index == py->index){
					double d = px->value - py->value;
					sum += d*d;
					++px;
					++py;
				}else{
					if(px->index > py->index){	
						sum += py->value * py->value;
						++py;
					}else{
						sum += px->value * px->value;
						++px;
					}
				}
			}
			while(px->index != -1){
				sum += px->value * px->value;
				++px;
			}
			while(py->index != -1){
				sum += py->value * py->value;
				++py;
			}
			d_output[y] = exp(-d_model_parameter.gamma*sum);
			break;
		}
	}
}

void CUDA_k_function(svm_node** SV, int modellen, const svm_parameter& param, const svm_node *x, double* output){
	const int threadPerBlock=256;
	const int blockPerGrid=(modellen+threadPerBlock-1)/threadPerBlock;
	if( d_SV==NULL ){
		checkCudaErrors(cudaMemcpyToSymbol(d_model_parameter, &param, sizeof(param)));
		checkCudaErrors(cudaMalloc((void ***)&d_SV, sizeof(struct svm_node*)*modellen));
		checkCudaErrors(cudaMemcpy(d_SV, SV, sizeof(struct svm_node*)*modellen, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void **)&d_output, sizeof(double)*modellen));
	}
    k_function<<<blockPerGrid, threadPerBlock>>>(x, d_SV, modellen, d_output);
    cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(output, d_output, sizeof(double)*modellen, cudaMemcpyDeviceToHost));
}

void CUDA_k_function_cleanup(){
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_SV));
    d_SV = NULL;
    d_output = NULL;
}
