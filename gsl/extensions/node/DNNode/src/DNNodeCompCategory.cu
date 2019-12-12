#ifndef CG_DNNODECOMPCATEGORY_CU
#define CG_DNNODECOMPCATEGORY_CU

#include "EdgeSetInput.h"

#define PRELIM_STATE DBL_MAX

void __global__ DNNode_kernel_initialize(
   double* output
   , double* gradient
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<EdgeSetInput, Array_Flat<int>::MemLocation::UNIFIED_MEM>* inputs
   #endif
   , double** weightedGradient
   , bool* ready
   , unsigned size
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
      output[index] = PRELIM_STATE;
      gradient[index] = PRELIM_STATE;
      ready[index] = false;
   }
}
void __global__ DNNode_kernel_update(
   double* output
   , double* gradient
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<EdgeSetInput, Array_Flat<int>::MemLocation::UNIFIED_MEM>* inputs
   #endif
   , double** weightedGradient
   , bool* ready
   , unsigned size
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
      auto end = inputs[index].end();
      if (!ready[index]) {
         for (auto iter=inputs[index].begin(); iter!=end; ++iter) {
            ready[index] = (*iter->inputArray)[iter->inputIndex] != PRELIM_STATE;
            if (!ready[index]) break;
         }
      }
      if (ready[index]) {
         output[index] = 0;
         for (auto iter=inputs[index].begin(); iter!=end; ++iter)
            //output[index] += (*iter->inputArray).getDataRef()[iter->inputIndex];
            output[index] += (*iter->inputArray)[iter->inputIndex];    
         gradient[index] = *(weightedGradient[index]);
      }
   }
}
#endif
