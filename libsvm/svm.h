#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 323

#ifdef __cplusplus
extern "C" {
#endif

extern int libsvm_version;

/**
 * svm_node
 * Struct
*/
struct svm_node{
	int index;
	double value;
};

/**
 * svm_problem
 * Struct
*/
struct svm_problem{
    int l; //!< Number of training vectors in \f$ \mathbb{R}^{n} \f$
    double *y; //!< Vector with corresponding values, i.e., fir C-SVC \f$y_{i}\in {-1,1}\f$
	struct svm_node **x;
};

/** Enum to save distint SVM formulation names (svm_type)*/
enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };
/** Enum to save distinct kernel types
 * linear:      \f$K(x_{i},x_{j}) = x_{i}^{T}x_{j}\f$.
 * polynomial:  \f$K(x_{i},x_{j}) = \left(\gamma x_{i}^{T}x_{j}+r\right)^{d}, \gamma > 0\f$.
 * radial basis function (RBF): \f$K(x_{i},x_{j}) = exp(-\gamma ||x_{i}-x_{j}||^{2}), \gamma > 0\f$.
 * sigmoid:     \f$K(x_{i},x_{j}) = tanh(\gamma x_{i}^{T}x_{j}+r)\f$.
 * precomputed:
 * \f$\gamma,r,\f$ and \f$d\f$ are kernel parameters.
*/
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };


/**
 * Struct to save distinct parameters used by each formulation of SVM:
 *
 * SVC: Support vector classification (two-clase and multi-class).
 * SVR: Support vector regression.
 * One-class SVM.
 *
 * SVC and SVR can also output probability estimates.
 *
*/
struct svm_parameter{
    int     svm_type;       //!< Integer that indicates the SVM type
    int     kernel_type;    //!< Integer that indicates the kernel type
    int     degree;         //!< for poly , in opencv use a double
    double  gamma;          //!< for poly/rbf/sigmoid
    double  coef0;          //!< for poly/sigmoid
    /* these are for training only */
    double  cache_size;     //!< in MB
    double  eps;            //!< stopping criteria
    double  C;              //!< for C_SVC, EPSILON_SVR and NU_SVR, C > 0 is the penalty parameter of the error term
    int     nr_weight;		//!< for C_SVC
    int     *weight_label;	//!< for C_SVC
    double  *weight;		//!< for C_SVC
    double  nu;             //!< for NU_SVC, ONE_CLASS, and NU_SVR
    double  p;              //!< for EPSILON_SVR
    int     shrinking;      //!< use the shrinking heuristics
    int     probability;    //!< do probability estimates
};

/**
 * @brief svm_model (for quadratic minimization problems)
 * C-support vector classification C-SVC
 * \f$\nu\f$-support vector classificaction  \f$\nu\f$-SVC
 * Distribution estimation one-class SVM
 * \f$\epsilon\f$-support vector regression \f$\epsilon\f$-SVR
 * \f$\nu\f$-support vector regression \f$\nu\f$-SVR
 *
 *
 */
struct svm_model{
	struct svm_parameter param;	/* parameter */
    int nr_class;	          	/* number of classes, = 2 in regression/one class svm */
    int l;			            /* total #SV */
	struct svm_node **SV;		/* SVs (SV[l]) */
    double **sv_coef;	        /* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
    double *rho;		        /* constants in decision functions (rho[k*(k-1)/2]) */
    double *probA;		        /* pariwise probability information */
	double *probB;
    int *sv_indices;            /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */
	/* for classification only */
    int *label;                 /* label of each class (label[k]) */
    int *nSV;                   /* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
    int free_sv;                /* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */
};

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
int svm_get_nr_sv(const struct svm_model *model);
double svm_get_svr_probability(const struct svm_model *model);

double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_free_model_content(struct svm_model *model_ptr);
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void svm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
