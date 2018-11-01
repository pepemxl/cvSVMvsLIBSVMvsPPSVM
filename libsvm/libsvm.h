/**
 * @file:   libsvm.h
 * @brief:  Wrapper interface for libSVM
 *
 */

#ifndef LIBSVM_H
#define	LIBSVM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <ctype.h>
#include <errno.h>
#include <locale.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

typedef float Treal;//!< Por el momento usare prencisión sencilla

namespace libsvm 
{
#include "svm.h"
}

using namespace libsvm;

/*LIBSVM* LIBSVM::getInstance() {
    static LIBSVM theInstance;
    return &theInstance;
}*/

/**
 * @brief      Class for libsvm.
 */
class LIBSVM {
private:
    struct svm_problem prob;
    struct svm_model* model;
    struct svm_node* x_space;

    bool trainingDataStructsUsed;
    bool predictionDataStructsUsed;
    
    int max_nr_attr;
    int predict_probability;

    char* line;
    unsigned long int max_line_len;

    void exit_input_error(int line_num) 
    {
        fprintf(stderr, "Wrong input format at line %d\n", line_num);
        exit(1);
    }
    /**
     * @brief      Method to read a line.
     *
     * @param      input  The input
     *
     * @return     A array of characteres.
     */
    char* readline(FILE *input)
    {
        int len;
        if(fgets(this->line,this->max_line_len,input) == NULL){
            return NULL;
        }
        /*while(strrchr(this->line,'\n' == NULL)){
            //max_line_len *= 2;
            this->line = (char *)realloc(this->line,this->max_line_len);
            len = (int)strlen(this->line);
            if(fgets(this->line+len,this->max_line_len-len,input) == NULL){
                break;
            }
            //std::cout << this->line << std::endl;
            //std::cout << max_line_len << std::endl;
        }*/
        //std::cout << this->line << std::endl;
        //std::cout << strlen(this->line) << std::endl;
        return this->line;
    }

public:
    /**
     * @brief      Constructs the object.
     */
    LIBSVM() : trainingDataStructsUsed(false), predictionDataStructsUsed(false)
    {
        line = NULL;                        
        max_nr_attr = 64;
        predict_probability = 1;
        param.cache_size = 512;             // in MB
        param.coef0 = 0.0;                  // for poly/sigmoid kernel
        param.degree = 3;                   // for poly kernel
        param.eps = 1e-3;                   // 0.001 stopping criteria
        param.gamma = 0;                    // for poly/rbf/sigmoid
        param.kernel_type = 0;              // libsvm::LINEAR; // type of kernel to use
        param.nr_weight = 0;                // for C_SVC
        param.nu = 0.5;                     // for NU_SVC, ONE_CLASS, and NU_SVR
        param.p = 0.1;                      // for EPSILON_SVR, epsilon in loss function?
        param.probability = predict_probability; // do probability estimates
        param.C = 0.01;                     // From paper, soft classifier
        param.shrinking = 0;                // use the shrinking heuristics, equals flag -h
        param.svm_type = C_SVC;             // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
        param.weight_label = NULL;          // for C_SVC
        param.weight = NULL;                // for C_SVC
        model = NULL;
        max_line_len = 2097152;
        //max_line_len = 1024;
        x_space = NULL;
    }

    /**
     * @brief      Destroys the object. Calls function freeMem.
     */
    virtual ~LIBSVM() 
    {
//        printf("Freeing used SVM memory\n");
//        if (this->predictionDataStructsUsed) {
//            free(x);
//        }
        this->freeMem();
    }

    /// Variables
    struct svm_parameter param;
    //static LIBSVM getInstance();

    /**
     * @brief      Gets the svm name.
     *
     * @return     The svm name.
     */
    const char* getSVMName() const 
    {
        return "libSVM";
    }
    
    /**
     * @brief      Free memory
     */
    void freeMem() 
    {
        if (this->trainingDataStructsUsed){
            free(prob.y);
            free(prob.x);
            free(x_space);
            free(line);            
        }
        svm_destroy_param(&this->param);
        svm_free_and_destroy_model(&model); // The model is used in training and prediction specific steps
    }

    /**
     * @brief      Reads a problem in a file in svmlight format.
     *
     * @param      filename  The filename
     * 
     * @todo There are an error in the detection of EOL, then have to be reviewed 
     * and corrected.
     */
    void read_problem(char *filename) 
    {
        setlocale(LC_NUMERIC,"C");
        setlocale(LC_ALL, "POSIX");
        int elements, max_index, inst_max_index, i, j;
        FILE *fp = fopen(filename, "r");
        char *endptr;
        char *idx, *val, *label;
        if(fp == NULL){
            fprintf(stderr, "can't open input file %s\n", filename);
            exit(1);
        }
        prob.l = 0;
        elements = 0;

        //max_line_len = 1024;
        max_line_len = 2097152;
        line = Malloc(char, max_line_len);
        while (readline(fp) != NULL) {
            char *p = strtok(line, " \t"); // label
            // features
            while (1) {
                p = strtok(NULL, " \t");
                if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                    break;
                ++elements;
            }
            ++elements;
            ++prob.l;
        }
        rewind(fp);
        prob.y = Malloc(double, prob.l);
        prob.x = Malloc(struct svm_node *, prob.l);
        x_space = Malloc(struct svm_node, elements);
        max_index = 0;
        j = 0;
        for(i = 0; i < prob.l; i++){
            inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
            readline(fp);
            prob.x[i] = &x_space[j];
            label = strtok(line, " \t\n");
            if (label == NULL) { // empty line
                printf("Empty line encountered!\n");
                exit_input_error(i + 1);
            }
            prob.y[i] = strtod(label, &endptr);
            if (endptr == label || *endptr != '\0') {
                printf("Wrong line ending encountered!\n");
                exit_input_error(i + 1);
            }
            while (1) {
                idx = strtok(NULL, ":");
                val = strtok(NULL, " \t");
                if (val == NULL)
                    break;
                errno = 0;
                x_space[j].index = (int) strtol(idx, &endptr, 10);
                if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index) {
                    printf("File input error at feature index encountered: %i, %i, %i, %i!\n", endptr == idx, errno != 0, *endptr != '\0', x_space[j].index <= inst_max_index);
                    exit_input_error(i + 1);                    
                } else {
                    inst_max_index = x_space[j].index;
                }
                errno = 0;
                x_space[j].value = std::strtod(val, &endptr);
//                printf("Value: '%f'\n", x_space[j].value);
                if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))) {
                    printf("File input error at feature value encountered: %i, %i, %i: '%s'\n",endptr == val, errno != 0, (*endptr != '\0' && !isspace(*endptr)), endptr);
                    exit_input_error(i + 1);
                }
                ++j;
            }
            if(inst_max_index > max_index){
                max_index = inst_max_index;
            }
            x_space[j++].index = -1;
        }

        if (param.gamma == 0 && max_index > 0) {
            param.gamma = 1.0 / max_index;
        }
        if (param.kernel_type == PRECOMPUTED) {
            for (i = 0; i < prob.l; i++) {
                if (prob.x[i][0].index != 0) {
                    fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
                    exit(1);
                }
                if ((int) prob.x[i][0].value <= 0 || (int) prob.x[i][0].value > max_index) {
                    fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
                    exit(1);
                }
            }
        }
        fclose(fp);
        this->trainingDataStructsUsed = true;
    }

    /**
     * Modelfile is saved to filesystem and contains the svm parameters used for training, which need to be retrieved for classification
     * Only makes sense after training was done
     * @param _modelFileName file name to save the model to
     */

    /**
     * @brief      Saves a model to file.
     *
     * @param[in]  _modelFileName  The model file name
     */
    void saveModelToFile(const std::string _modelFileName)
    {
        if (svm_save_model(_modelFileName.c_str(), this->model)){
            fprintf(stderr, "Error: Could not save model to file %s\n", _modelFileName.c_str());
            exit(EXIT_FAILURE);
        }
    }

    /**
     * Function was unit tested, can be assumed libSVM model is correctly loaded from file
     * @param _modelFileName
     */

    /**
     * @brief      Loads a model from file.
     *
     * @param[in]  _modelFileName  The model file name
     */
    void loadModelFromFile(const std::string _modelFileName)
    {
        this->freeMem();
        /// @WARNING: This is really important, ROS seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
        // Do not use the system locale setlocale(LC_ALL, "C");
        setlocale(LC_NUMERIC,"C");
        setlocale(LC_ALL, "POSIX");
        /// @TODO Test if this works as intended
        if (this->predictionDataStructsUsed && model != NULL) {
//            svm_free_and_destroy_model(&model);
            delete model; // clear the previous model and deallocate/free the occupied memory
            model = NULL; // set the pointer to NULL
        }
        printf("Loading model from file '%s'\n", _modelFileName.c_str());
        model = svm_load_model(_modelFileName.c_str());
        if (model == NULL) {
            printf("Loading of SVM model failed!\n");
            exit(EXIT_FAILURE);
        }

        int classNr = svm_get_nr_class(model);
        int labels[classNr];
        for (int i = 0; i < classNr; ++i) {
                labels[i] = 0;
        }
        svm_get_labels(model, labels);
        double b = -(model->rho[0]);
        double probA = (model->probA[0]);
        double probB = (model->probB[0]);
        
        int kernelType = model->param.kernel_type;
        printf("Loaded model: SVM type %d, Kernel type %d, %d classes: labels %d, %d, #SVs %d, bias b %3.5f, probA %3.5f, probB %3.5f\n", svm_get_svm_type(model), kernelType, classNr, labels[0], labels[1], model->l, b, probA, probB);
        this->predictionDataStructsUsed = true;
//        exit(EXIT_SUCCESS);
    }
    /**
     * @brief      Gets the problem.
     *
     * @return     The problem.
     */
    svm_problem getProblem()
    {
        return this->prob;
    }
    /**
     * @brief      Predict label
     *
     * @param      _sample        The sample
     * @param      _probEstimate  The prob estimate
     *
     * @return     result of prediction
     */
    float predictLabel(svm_node* _sample, double* _probEstimate)
    {
        float predict_label = svm_predict_probability(model, _sample, _probEstimate);
        return predict_label;
    }
    /**
     * @brief      { After read in the training samples from a file, set parameters for training and call training procedure}
     */
    void train()
    {
        model = svm_train(&prob, &param);
        trainingDataStructsUsed = true;
    }
    /**
     * @brief      Gets the single detecting vector. Generates a single detecting feature vector (vec1) from the trained support vectors, for use e.g. with the HOG algorithm
     * vec1 = sum_1_n (alpha_y*x_i). (vec1 is a 1 x n column vector. n = feature vector length )
     *
     * @param      singleDetectorVector         resulting single detector vector for use in openCV HOG
     * @param      singleDetectorVectorIndices  vector containing indices of features inside singleDetectorVector
     */
    void getSingleDetectingVector(std::vector<Treal>& singleDetectorVector, std::vector<unsigned int>& singleDetectorVectorIndices) 
    {
        // Now we use the trained svm to retrieve the single detector vector
        // COMENTAR para aplicacion
        //printf("Calculating single detecting feature vector out of support vectors (may take some time)\n");
        singleDetectorVector.clear();
        singleDetectorVectorIndices.clear();
        // COMENTAR para aplicacion
        //printf("Total number of support vectors: %d \n", model->l);
        // Walk over every support vector and build a single vector
        for (unsigned long ssv = 0; ssv < model->l; ++ssv) { // Walks over available classes (e.g. +1, -1 representing positive and negative training samples)
            // Retrieve the current support vector from the training set
            svm_node* singleSupportVector = model->SV[ssv]; // Get next support vector ssv==class, 2nd index is the component of the SV
            // sv_coef[i] = alpha[i]*sign(label[i]) = alpha[i] * y[i], where i is the training instance, y[i] in [+1,-1]
            double alpha = model->sv_coef[0][ssv];
            int singleVectorComponent = 0;
            while (singleSupportVector[singleVectorComponent].index != -1) { // index=-1 indicates the end of the array
                if (ssv == 0) { // During first loop run determine the length of the support vectors and adjust the required vector size
                    singleDetectorVector.push_back(singleSupportVector[singleVectorComponent].value * alpha);
                    singleDetectorVectorIndices.push_back(singleSupportVector[singleVectorComponent].index); // Holds the indices for the corresponding values in singleDetectorVector, mapping from singleVectorComponent to singleSupportVector[singleVectorComponent].index!
                } else {
                    if (singleVectorComponent > singleDetectorVector.size()) { // Catch oversized vectors (maybe from differently sized images?)
                        printf("Warning: Component %d out of range, should have the same size as other/first vector\n", singleVectorComponent);
                    } else
                        singleDetectorVector.at(singleVectorComponent) += (singleSupportVector[singleVectorComponent].value * alpha);
                }
                singleVectorComponent++;
            }
        }
    }
    /**
     * @brief      Return model detection threshold / bias
     *
     * @return     detection threshold / bias
     */
    double getThreshold() const 
    {
        return model->rho[0];
    }

};




#endif	/* LIBSVM_H */
