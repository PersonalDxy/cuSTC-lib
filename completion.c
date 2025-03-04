//includes
#include "als.cuh"
#include "sgd.cuh"
#include "ccd.cuh"
#include "sptensor.h"
#include "completion.h"
#include "matrixprocess.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

//all the tc_* in completion can be paralleled

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Predict a value for a three-way tensor.
*
* @param test The test tensor which gives us the model rows.
* @param index The nonzero to draw the row indices from. test[0][index] ...
*
* @return The predicted value.
*/
static inline double p_predict_val3(
    idx_t nmodes,
    sptensor_t * test,
    ordi_matrix ** mats,
    idx_t const index)
{
  double est = 0;
  assert(nmodes == 3);
  idx_t const i = test->ind[0][index];
  idx_t const j = test->ind[1][index];
  idx_t const k = test->ind[2][index];

  double * A = mats[0]->values + (i * DEFAULT_NFACTORS);
  double * B = mats[1]->values + (j * DEFAULT_NFACTORS);
  double * C = mats[2]->values + (k * DEFAULT_NFACTORS);

  idx_t f;
  for(f = 0; f < DEFAULT_NFACTORS; ++f) {
    est += A[f] * B[f] * C[f];
  }

  return est;
}


/**
* @brief Predict a value for a three-way tensor when the model uses column-major
*        matrices.
*
* @param model The column-major model to use for the prediction.
* @param test The test tensor which gives us the model rows.
* @param index The nonzero to draw the row indices from. test[0][index] ...
*
* @return The predicted value.
*/
static inline double p_predict_val3_col(
    idx_t nmodes,
    sptensor_t * test,
    ordi_matrix ** mats,
    idx_t const index)
{
  double est = 0;
  
  assert(nmodes == 3);

  idx_t const i = (test->ind)[0][index];
  idx_t const j = (test->ind)[1][index];
  idx_t const k = (test->ind)[2][index];

  idx_t const I = test->dims[0];
  idx_t const J = test->dims[1];
  idx_t const K = test->dims[2];

  double * A = mats[0]->values;
  double * B = mats[1]->values;
  double * C = mats[2]->values;

  idx_t f;
  for(f=0; f < DEFAULT_NFACTORS; ++f) {
    est += A[i+(f*I)] * B[j+(f*J)] * C[k+(f*K)];
  }

  return est;
}


/**
* @brief Print some basic statistics about factorization progress.
*
* @param epoch Which epoch we are on.
* @param loss The sum-of-squared loss.
* @param rmse_tr The RMSE on the training set.
* @param rmse_vl The RMSE on the validation set.
* @param ws Workspace, used for timing information.
*/
static void p_print_progress(
    idx_t const epoch,
    double const loss,
    double const rmse_tr,
    double const rmse_vl
    )
{
  printf("epoch:%d   loss: %0.5e   "
      "RMSE-tr: %0.5e   RMSE-vl: %0.5e \n",
      epoch, loss, rmse_tr, rmse_vl);
}




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

double tc_rmse(
    sptensor_t * test,
    ordi_matrix ** mats,
    int algorithm_index
    )
{
return sqrt((tc_loss_sq(test, mats, algorithm_index)) / (test->nnz));

}



double tc_mae(
    sptensor_t * test,
    ordi_matrix ** mats,
    int algorithm_index
    )
{
  double loss_obj = 0.;
  double * test_vals = test->vals;
  idx_t x;

  if(algorithm_index == 2) { //if the algorithm is CCD+
    for(x=0; x < test->nnz; ++x) {
       double predicted = tc_predict_val_col(test, mats, x);
       loss_obj += fabs(test_vals[x] - predicted);
    }
    } else {
    
      for(x=0; x < test->nnz; ++x) {
        double  predicted = tc_predict_val(test,mats,x);
        loss_obj += fabs(test_vals[x] - predicted);
      }
    }


   return loss_obj / test->nnz;

}



//损失函数 算的是 test中的坐标实际值和根据mats的预测值算出来的差平方
double tc_loss_sq(
    sptensor_t * test,
    ordi_matrix ** mats,
    int algorithm_index
    )
{
  double loss_obj = 0.;
  double * test_vals = test->vals;
  idx_t x;

  if(algorithm_index == 2) {
    for(x=0; x < test->nnz; ++x) {
      double const err = test_vals[x] - tc_predict_val_col(test, mats, x);
      loss_obj += err * err;
      }
    } else {
    for(x=0; x < test->nnz; ++x) {
      double temp = tc_predict_val(test, mats, x);
      double const err = test_vals[x] - temp;
      // printf("index: %d, val: %d, predict: %d\n", x, test_vals[x], temp);
      loss_obj += err * err;
      }
    }

return loss_obj;
}


//算因子矩阵中所有值的平方和？正则化吧
double tc_frob_sq(
    idx_t nmodes,
    double regularization_index,
    ordi_matrix ** mats)
{
  double reg_obj = 0.;
  idx_t x,m;

  for(m=0; m < nmodes; ++m) {
      double accum = 0;
      idx_t const nrows = mats[m]->I;
      double * mat = mats[m]->values;
      for(x=0; x < nrows * DEFAULT_NFACTORS; ++x) {
        accum += mat[x] * mat[x];
      }
      reg_obj += regularization_index * accum;
    }
   

  assert(reg_obj > 0);
  return reg_obj;
}



double tc_predict_val(
    sptensor_t * test,
    ordi_matrix ** mats,
    idx_t const index
    )
{
  if(test->nmodes == 3) {
    return p_predict_val3(test->nmodes, test, mats, index);
  }

  /*idx_t const DEFAULT_NFACTORS = model->rank;

  // initialize accumulation of each latent factor with the first row //
  idx_t const row_id = test->ind[0][index];
  double const * const init_row = model->factors[0] + (row_id * DEFAULT_NFACTORS);
  for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
    buffer[f] = init_row[f];
  }

  // now multiply each factor by A(i,:), B(j,:) ... //
  idx_t const nmodes = model->nmodes;
  for(idx_t m=1; m < nmodes; ++m) {
    idx_t const row_id = test->ind[m][index];
    double const * const row = model->factors[m] + (row_id * DEFAULT_NFACTORS);
    for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
      buffer[f] *= row[f];
    }
  }

  // finally, sum the factors to form the final estimated value //
  double est = 0;
  for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
    est += buffer[f];
  }
  

  return est;*/
}


double tc_predict_val_col(
    sptensor_t * test,
    ordi_matrix ** mats,
    idx_t const index
    )
{
  if(test->nmodes == 3) {
    return p_predict_val3_col(test->nmodes, test, mats, index);
  }

  /*idx_t const DEFAULT_NFACTORS = model->rank;

  //initialize accumulation of each latent factor with the first row //
  idx_t const row_id = test->ind[0][index];
  double const * const init_row = model->factors[0] + (row_id * DEFAULT_NFACTORS);
  for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
    buffer[f] = model->factors[0][row_id + (f * model->dims[0])];
  }

  // now multiply each factor by A(i,:), B(j,:) ... //
  idx_t const nmodes = model->nmodes;
  for(idx_t m=1; m < nmodes; ++m) {
    idx_t const row_id = test->ind[m][index];
    for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
      buffer[f] *= model->factors[m][row_id + (f * model->dims[m])];
    }
  }

  // finally, sum the factors to form the final estimated value //
  double est = 0;
  for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
    est += buffer[f];
  } 

  return est; */
}


bool tc_converge(
    sptensor_t * train,
    sptensor_t * validate,
    ordi_matrix ** mats,
    ordi_matrix ** best_mats,
    int algorithm_index,
    double const loss,
    double const frobsq,
    idx_t const epoch,
    idx_t nmodes,
    double * best_rmse,
    double * tolerance,
    idx_t * nbadepochs,
    idx_t * bestepochs,
    idx_t * max_badepochs
    )
{
   printf("maxepoch: %d\n", *max_badepochs);
   double const train_rmse = sqrt(loss / train->nnz);
   idx_t m;

   double converge_rmse = train_rmse;

  /* optionally compute validation */
  double val_rmse = 0.;
  if(validate != NULL) {
    val_rmse = tc_rmse(validate, mats, algorithm_index);

    /* base convergence on validation */
    converge_rmse = val_rmse;
  }

p_print_progress(epoch, loss, train_rmse, val_rmse);

  // to be optimized, first just set the fixed converge paramemters, and need to add workspace
  bool converged = false;

  if(converge_rmse - *(best_rmse) < -*(tolerance)) {
    *nbadepochs = 0;
    *best_rmse = converge_rmse;
    *bestepochs = epoch;

    /* save the best model, it is removed this time,  */
    for(m=0; m < train->nmodes; ++m) {
      matrix_copy(mats[m], best_mats[m]);
      /* TODO copy globmats too*/
    }
  } else {
    *(nbadepochs) = *(nbadepochs) + 1;
    if(*nbadepochs == *max_badepochs) {
      converged = true;
    }
    /*for(m=0; m < train->nmodes; ++m) {
      matrix_copy(best_mats[m], mats[m]);
      // TODO copy globmats too
    }*/
  }

  /* TODO: check for time limit 
  if(ws->max_seconds > 0 && ws->tc_time.seconds >= ws->max_seconds) {
    converged = true;
  }*/

  /*if(!converged) {
    timer_start(&ws->tc_time);
  }*/
  return converged;
}


/**
* @brief The function for testing the converge and toher transferred function, including output under COO format. Only contains SGD and CCD.
*/
/*void tc_main_coo(sptensor_t* train, sptensor_t* validation, sptensor_t* test, int algorithm_index, idx_t nmodes)
{
    double regularization_index = 0;
    //idx_t nmodes = train->nmodes;

    //intialize the factor matrix
    ordi_matrix * mats[MAX_NMODES+1];
    best_mats=(ordi_matrix**)malloc((nmodes)*sizeof(ordi_matrix*));
    idx_t m = 0;
    // allocate factor matrices 
    idx_t maxdim = train->dims[argmax_elem(train->dims, nmodes)];
    for(m=0; m < nmodes; ++m) {
      mats[m] = (ordi_matrix*) matrix_randomize(train->dims[m], DEFAULT_NFACTORS);
    }
    mats[MAX_NMODES] = matrix_alloc(maxdim, DEFAULT_NFACTORS);

    //initialize the parameters    
    double best_rmse = BEST_RMSE;
    double tolerance = TOLERANCE;
    idx_t nbadepochs = NBADEPOCHS;
    idx_t bestepochs = BEST_EPOCH;
    idx_t max_badepochs = MAX_BADEPOCHS;

    switch (algorithm_index)
    {
    case 1:
        //may need format transformation
        regularization_index = SGD_REGULARIZATION;
        //learning rate
        double learning_rate = LEARN_RATE;
        tc_sgd(train, validation, test, mats, best_mats, algorithm_index, regularization_index, learning_rate, &best_rmse, &tolerance, &nbadepochs, &bestepochs, &max_badepochs);
        break;    
    
    default:
        regularization_index = CCD_REGULARIZATION;
        tc_ccd(train, validation, test, mats, best_mats, algorithm_index, regularization_index,  &best_rmse, &tolerance, &nbadepochs, &bestepochs, &max_badepochs);
        break;
    }

    #ifdef WITHOUTPUT
    //TO DO output the factor matrices
    #endif
    
    for(m=0;m<train->nmodes;m++)
    {
        matrix_free(mats[m]);
    }
    matrix_free(mats[MAX_NMODES]);
    free(best_mats);
} */


void write_matrices_to_file(const char *filename, ordi_matrix **mats, int num_mats) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for writing");
        exit(EXIT_FAILURE);
    }

    // 写入矩阵的数量
    fwrite(&num_mats, sizeof(int), 1, file);

    for (int i = 0; i < num_mats; i++) {
        ordi_matrix *mat = mats[i];

        // 写入矩阵属性
        fwrite(&mat->I, sizeof(idx_t), 1, file);
        fwrite(&mat->J, sizeof(idx_t), 1, file);
        fwrite(&mat->row_col_flag, sizeof(int), 1, file);

        // 写入矩阵值
        size_t num_values = mat->I * mat->J;
        fwrite(mat->values, sizeof(double), num_values, file);
    }

    fclose(file);
}

/**
 * @brief The main function for tensor completion under MBLC(in ciss.h)
*/
void tc_main_ciss(sptensor_t* traina, sptensor_t* trainb, sptensor_t* trainc, sptensor_t* validation, sptensor_t* test, int algorithm_index)
{
    
    double regularization_index = 0;
    srand(time(NULL));
    //intialize the factor matrix
    ordi_matrix ** mats;
    ordi_matrix ** best_mats;
    idx_t nmodes = traina->nmodes;
    mats=(ordi_matrix**)malloc((MAX_NMODES)*sizeof(ordi_matrix*));//这个是最大的因子矩阵
    best_mats=(ordi_matrix**)malloc((nmodes)*sizeof(ordi_matrix*));//这个是根据张量维度创建的因子矩阵数量
    idx_t m = 0;
    
    /* allocate factor matrices */
    #ifdef CISS_DEBUG
      printf("nmodes %d\n",nmodes);
    #endif
    idx_t maxdim = traina->dims[argmax_elem(traina->dims, nmodes)];//维度最大的那个维度
    for(m=0; m < nmodes; m++) {
      #ifdef CISS_DEBUG
      printf("the correct dimension in mode %d is %d\n",m,traina->dims[m]);
      #endif
      mats[m] = (ordi_matrix*) matrix_randomize(traina->dims[m], (idx_t)DEFAULT_NFACTORS);//开辟因子矩阵空间并随机初始化 秩默认是16
      best_mats[m] = (ordi_matrix*) matrix_randomize(traina->dims[m], (idx_t)DEFAULT_NFACTORS);
      matrix_copy(mats[m],best_mats[m]);//这为什么要将mats赋值给best-mats
      #ifdef CISS_DEBUG
      printf("the new dimension in mode %d is %d\n",m,mats[m]->I);
      #endif
    }
    mats[MAX_NMODES-1] = matrix_alloc(maxdim, (idx_t)DEFAULT_NFACTORS);

    //initialize the parameters    
    double best_rmse = (double)BEST_RMSE;
    double tolerance = (double)TOLERANCE;
    idx_t nbadepochs = NBADEPOCHS;
    idx_t bestepochs = BEST_EPOCH;
    idx_t max_badepochs = MAX_BADEPOCHS;

    switch (algorithm_index)
    {
    case 0:
        //may need format transformation
        regularization_index = (double)ALS_REGULARIZATION;
        tc_als(traina, trainb, trainc, validation, test, mats, best_mats, algorithm_index,  regularization_index, &best_rmse, &tolerance, &nbadepochs, &bestepochs, &max_badepochs);
        break;

    case 1:
        //may need format transformation
        regularization_index = (double)SGD_REGULARIZATION;
        //learning rate
        double learning_rate = (double)LEARN_RATE;
        tc_sgd(traina, trainb, trainc, validation, test, mats, best_mats, algorithm_index,  regularization_index, learning_rate, &best_rmse, &tolerance, &nbadepochs, &bestepochs, &max_badepochs);
        break;    
    
    default:
        regularization_index = (double)CCD_REGULARIZATION;
        tc_ccd(traina, trainb, trainc, validation, test, mats, best_mats, algorithm_index, regularization_index,  &best_rmse, &tolerance, &nbadepochs, &bestepochs, &max_badepochs);
        break;
    }

    #ifdef WITHOUTPUT
    //TO DO output the factor matrices
    #endif

    write_matrices_to_file("factor_matrix.json", mats, 3);
    
    for(m=0;m<nmodes;m++)
    {
        matrix_free(mats[m]);
    }
    for(m=0;m<nmodes;m++)
    {
        matrix_free(best_mats[m]);
    }
    matrix_free(mats[MAX_NMODES-1]);
    free(mats);
    free(best_mats);
}