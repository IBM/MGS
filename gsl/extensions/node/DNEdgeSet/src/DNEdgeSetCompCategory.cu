#ifndef CG_DNEDGESETCOMPCATEGORY_CU
#define CG_DNEDGESETCOMPCATEGORY_CU

#include <stdio.h>
//#include "helper_cuda.h"
#include "rndm.h"
#include "TransferFunction.h"
#include <math.h>

#define PRELIM_STATE DBL_MAX
#define SMALL_NUMBER 0.00000001

__device__ double tanh(double x)
{
  return 2/(1+exp(-2*x))-1; 
}

__device__ double dtanh(double _tanh_) {
	//This function is designed to receive tanh as an argument
	return 1.0 - _tanh_ * _tanh_;
}
__device__ double relu(double input) {
	return ( (input>0) ? input : 0.0 );
}
__device__ double drelu(double _relu_) {
	return ( (_relu_>0) ? 1.0 : 0.0 );
}

typedef double (*transfer_fptr_t)( double);
typedef double (*dervtransfer_fptr_t)( double);

#define OPTION_1 1
#define OPTION_2 2
#define OPTION_3 3
#define DEVICE_FNC_OPTION OPTION_2

/* LIMIT: list of transfer function must be known at compile time */
#if DEVICE_FNC_OPTION == OPTION_1
//OPTION 1
__constant__ transfer_fptr_t d_fListTransfer[NUM_TRANSFER_FUNC]={tanh, relu};
__constant__ dervtransfer_fptr_t d_fListDervTransfer[NUM_TRANSFER_FUNC]={dtanh, drelu};
//__host__ __device__ double Sum(double v) {return v+1;}
//__host__ __device__ double Subtract(double v) {return v-1;}
//__host__ __device__ double Multiply(double v) {return v*2;}
//__constant__ transfer_fptr_t d_fListTransfer[NUM_TRANSFER_FUNC]={Sum, Subtract};
#endif

/* LIMIT: list of transfer function must be known at compile time */
#if DEVICE_FNC_OPTION == OPTION_2
//OPTION 2
__device__ transfer_fptr_t d_fListTransfer[NUM_TRANSFER_FUNC] = {tanh, relu };
__device__ dervtransfer_fptr_t d_fListDervTransfer[NUM_TRANSFER_FUNC]={dtanh, drelu};
#endif

/* specific individual pointer to device function */
#if DEVICE_FNC_OPTION == OPTION_3
//OPTION 3: need static device pointer (so that CUDA generate address of a device function)
__device__ transfer_fptr_t d_fListTransfer[NUM_TRANSFER_FUNC];
__device__ dervtransfer_fptr_t d_fListDervTransfer[NUM_TRANSFER_FUNC];
// Declare device side function pointers.  We retrieve them later with
// cudaMemcpyFromSymbol to set our function tables above in some
// particular order specified at runtime.
#endif
__device__ transfer_fptr_t p_Tanh = tanh;
__device__ transfer_fptr_t p_ReLu = relu;
__device__ dervtransfer_fptr_t p_dTanh = dtanh;
__device__ dervtransfer_fptr_t p_dReLu = drelu;

// Allocate host side tables to mirror the device side, and later, we
// fill these tables with the function pointers.  This lets us send
// the pointers to the kernel on invocation, as a method of choosing
// which function to run.
transfer_fptr_t h_fListTransfer[NUM_TRANSFER_FUNC];
dervtransfer_fptr_t h_fListDervTransfer[NUM_TRANSFER_FUNC];

// Copy the pointers from the function tables to the host side
void setupFunctionTables()
{
   std::cout << "Evoking setupFunctionTables() \n"; 
   //put the GPU-side address into host-side variable
#if DEVICE_FNC_OPTION == OPTION_1 || DEVICE_FNC_OPTION == OPTION_2
   checkCudaErrors(cudaMemcpyFromSymbol(h_fListTransfer, d_fListTransfer, NUM_TRANSFER_FUNC * sizeof(transfer_fptr_t)));
   checkCudaErrors(cudaMemcpyFromSymbol(h_fListDervTransfer, d_fListDervTransfer, NUM_TRANSFER_FUNC * sizeof(dervtransfer_fptr_t)));
#endif
#if DEVICE_FNC_OPTION == OPTION_3
   {
      checkCudaErrors(cudaMemcpyFromSymbol(&h_fListTransfer[TANH], p_Tanh, sizeof(transfer_fptr_t)));
      checkCudaErrors(cudaMemcpyFromSymbol(&h_fListDervTransfer[TANH], p_dTanh, sizeof(dervtransfer_fptr_t)));
      checkCudaErrors(cudaMemcpyFromSymbol(&h_fListTransfer[RELU], p_ReLu, sizeof(transfer_fptr_t)));
      checkCudaErrors(cudaMemcpyFromSymbol(&h_fListDervTransfer[RELU], p_dReLu, sizeof(dervtransfer_fptr_t)));
      // Dynamically assign the function table.
      // Copy the function pointers to their appropriate locations according to the enum
      //checkCudaErrors(cudaMemcpyFromSymbol(&h_transfer_fptr_table[SOBEL_FILTER], pComputeSobel, sizeof(transfer_fptr_t)));
      //checkCudaErrors(cudaMemcpyFromSymbol(&h_transfer_fptr_table[SOBEL_FILTER], pComputeSobel, sizeof(transfer_fptr_t)));
      //checkCudaErrors(cudaMemcpyFromSymbol(&h_transfer_fptr_table[BOX_FILTER], pComputeBox, sizeof(transfer_fptr_t)));

      // do the same for the point function, where the 2nd function is NULL ("no-op" filter, skipped in kernel code)
      /*checkCudaErrors(cudaMemcpyFromSymbol(&h_pointFunction_table[THRESHOLD_FILTER], pComputeThreshold, sizeof(pointFunction_t)));*/
      /*h_pointFunction_table[NULL_FILTER] = NULL;*/

      // now copy the function tables back to the device, so if we wish we can use an index into the table to choose them
      // We have now set the order in the function table according to our enum.
      checkCudaErrors(cudaMemcpyToSymbol(d_fListTransfer, h_fListTransfer, sizeof(transfer_fptr_t)*NUM_TRANSFER_FUNC));
      checkCudaErrors(cudaMemcpyToSymbol(d_fListDervTransfer, h_fListDervTransfer, sizeof(dervtransfer_fptr_t)*NUM_TRANSFER_FUNC));
   }
#endif
}
void __global__ DNEdgeSet_kernel_initialize(
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* weights
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* weights
   , int* weights_start_offset
   , int* weights_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* weights
   , int weights_max_elements
   , int* weights_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* weights
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* deltaWeights
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* deltaWeights
   , int* deltaWeights_start_offset
   , int* deltaWeights_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* deltaWeights
   , int deltaWeights_max_elements
   , int* deltaWeights_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* deltaWeights
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* deltaWeightsSquared
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* deltaWeightsSquared
   , int* deltaWeightsSquared_start_offset
   , int* deltaWeightsSquared_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* deltaWeightsSquared
   , int deltaWeightsSquared_max_elements
   , int* deltaWeightsSquared_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* deltaWeightsSquared
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* weightedOutputs
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* weightedOutputs
   , int* weightedOutputs_start_offset
   , int* weightedOutputs_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* weightedOutputs
   , int weightedOutputs_max_elements
   , int* weightedOutputs_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* weightedOutputs
   //need more info here
   #endif

   , double* weightedGradient
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* echoes
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* echoes
   , int* echoes_start_offset
   , int* echoes_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* echoes
   , int echoes_max_elements
   , int* echoes_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* echoes
   //need more info here
   #endif

   , unsigned* echoIndex
   , double* biasCorrectionW
   , double* biasCorrectionS
   , double** input
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double*, Array_Flat<int>::MemLocation::UNIFIED_MEM>* gradients
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double** gradients
   , int* gradients_start_offset
   , int* gradients_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double** gradients
   , int gradients_max_elements
   , int* gradients_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double* >* gradients
   //need more info here
   #endif

   , bool* readyForward
   , bool* readyBackward
   , bool* momentum
   , bool* rmsprop
   , unsigned size
   , double beta
   , double alpha
   , double eta
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
      assert(0);
   }
}
void __global__ DNEdgeSet_kernel_update(
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* weights
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* weights
   , int* weights_start_offset
   , int* weights_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* weights
   , int weights_max_elements
   , int* weights_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* weights
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* deltaWeights
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* deltaWeights
   , int* deltaWeights_start_offset
   , int* deltaWeights_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* deltaWeights
   , int deltaWeights_max_elements
   , int* deltaWeights_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* deltaWeights
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* deltaWeightsSquared
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* deltaWeightsSquared
   , int* deltaWeightsSquared_start_offset
   , int* deltaWeightsSquared_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* deltaWeightsSquared
   , int deltaWeightsSquared_max_elements
   , int* deltaWeightsSquared_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* deltaWeightsSquared
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* weightedOutputs
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* weightedOutputs
   , int* weightedOutputs_start_offset
   , int* weightedOutputs_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* weightedOutputs
   , int weightedOutputs_max_elements
   , int* weightedOutputs_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* weightedOutputs
   //need more info here
   #endif

   , double* weightedGradient
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* echoes
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* echoes
   , int* echoes_start_offset
   , int* echoes_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* echoes
   , int echoes_max_elements
   , int* echoes_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* echoes
   //need more info here
   #endif

   , unsigned* echoIndex
   , double* biasCorrectionW
   , double* biasCorrectionS
   , double** input
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double*, Array_Flat<int>::MemLocation::UNIFIED_MEM>* gradients
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double** gradients
   , int* gradients_start_offset
   , int* gradients_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double** gradients
   , int gradients_max_elements
   , int* gradients_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double* >* gradients
   //need more info here
   #endif
   , bool* readyForward
   , bool* readyBackward
   , bool* momentum
   , bool* rmsprop
   , unsigned size
   , int* fncIndex
   , double beta
   , double alpha
   , double eta
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
     if (!readyForward[index])
       readyForward[index] = *(input[index]) != PRELIM_STATE;

     auto gend = gradients[index].end();
     if (readyForward[index]) {
         auto witer=weights[index].begin();
         auto woiter=weightedOutputs[index].begin();
         auto woend=weightedOutputs[index].end();

         //double transferInput = transferFunction.transfer(*input);
         double transferInput = (*d_fListTransfer[fncIndex[index]])(*input[index]);

         for (; woiter!=woend; ++woiter, ++witer) {
            *woiter = *witer * transferInput;
         }
         if (!readyBackward[index]) {
            for (auto giter=gradients[index].begin(); giter!=gend; ++giter) {
               readyBackward[index] = **giter != PRELIM_STATE;
               if (!readyBackward[index]) {	  
                  /* DISCUSS: dynamic size inside kernel? */
                  echoes[index].push_back(transferInput);	  
                  break;
               }
            }
         }
         if (readyBackward[index]) {
            double dow = 0;
            //assert(getSimulation().getIteration()>0);
            auto diter=deltaWeights[index].begin();
            auto siter=deltaWeightsSquared[index].begin();

            witer=weights[index].begin();      
            for (auto giter=gradients[index].begin(); giter!=gend; ++giter, ++witer, ++diter, ++siter) {
               dow +=  *witer * **giter;

               double deltaWeight = echoes[index][echoIndex[index]] * **giter;
               double update = eta;

               if (rmsprop[index])
                  *siter = ( (1-beta) * deltaWeight * deltaWeight + beta * *siter ) / (1.0 - biasCorrectionS[index]);

               if (momentum[index]) {
                  *diter = ( (1-alpha) * deltaWeight + alpha * *diter ) / (1.0 - biasCorrectionW[index]);
                  update *=  *diter;
               }
               else
                  update *= deltaWeight;

               if (rmsprop[index])
                  update /= sqrt(*siter + SMALL_NUMBER);

               *witer += update;
            }

            if (momentum[index]) biasCorrectionW[index] *= alpha;
            if (rmsprop[index]) biasCorrectionS[index] *= beta;

            weightedGradient[index] = dow * (*d_fListDervTransfer[fncIndex[index]])(echoes[index][echoIndex[index]]);
            echoes[index][echoIndex[index]] = transferInput;
            if (++echoIndex[index] == echoes[index].size()) echoIndex[index] = 0; 
         }
     }

   }
}
#endif
