#ifndef LIFENODECOMPCATEGORY_CU
#define LIFENODECOMPCATEGORY_CU

void __global__ LifeNode_kernel_initialize(
     int* value,
     int* publicValue,
   //#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   //  ShallowArray_Flat<int*, Array_Flat<int>::MemLocation::UNIFIED_MEM>* neighbors,  
   //#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   //  int** neighbors,  
   //#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   //  int** neighbors,  
   //#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   //  int** neighbors,  
   //#endif
     //RNG& rng
     unsigned size,
     int tooSparse,
     int tooCrowded
     ) 
{
   int index =  blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size)
   {
      publicValue[index]=value[index];
   }
}
void __global__ LifeNode_kernel_update(
     int* value,
     int* publicValue,
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
     ShallowArray_Flat<int*, Array_Flat<int>::MemLocation::UNIFIED_MEM>* neighbors,  
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
     int** neighbors,  
     int* neighbors_start_offset,  
     int* neighbors_num_elements,  
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
     int** neighbors,  
     int neighbors_max_elements,
     int* neighbors_num_elements,  
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
     int** neighbors,  
   #endif
     // //RNG& rng
      unsigned size
     , int tooSparse 
     , int tooCrowded
      ) 
{
   int index =  blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size)
   {
      int neighborCount=0;
      /* TUAN TODO find out the bug in here */
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
      ShallowArray_Flat<int*>::iterator end = neighbors[index].end();
      for (auto iter=neighbors[index].begin(); iter!=end; ++iter) {
         neighborCount += **iter;
      }
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
      auto um_neighbors_from = neighbors_start_offset[index];
      auto um_neighbors_to = um_neighbors_num_elements[index]-1;
      for (auto idx = um_neighbors_from; idx < um_neighbors_to; ++idx) {
         neighborCount += *(neighbors[idx]);
      }
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
      auto um_neighbors_from = index * neighbors_max_elements;
      auto um_neighbors_to = um_neighbors_from + neighbors_num_elements[index];
      for (auto idx = um_neighbors_from; idx < um_neighbors_to; ++idx) {
         neighborCount += *(neighbors[idx]);
      }
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   #endif

      if (neighborCount<= tooSparse || neighborCount>= tooCrowded) {
         value[index]=0;
      }
      else {
         value[index]=1;
      }
   }
}

void __global__ LifeNode_kernel_copy(
     int* value,
     int* publicValue,
     float* weight,
     float* publicWeight,
     unsigned size
     ) 
{
   int index =  blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size)
   {
      publicValue[index]=value[index];
      publicWeight[index]=weight[index];
   }
}

void __global__ LifeNode_kernel_updateWeight(
   int* value
   , int* publicValue
   , float* weight
   , float* publicWeight
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<int*, Array_Flat<int>::MemLocation::UNIFIED_MEM>* neighbors
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , int** neighbors
   , int* neighbors_start_offset
   , int* neighbors_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , int** neighbors
   , int neighbors_max_elements
   , int* neighbors_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< int* >* neighbors
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<float*, Array_Flat<int>::MemLocation::UNIFIED_MEM>* neighborsWeight
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , float** neighborsWeight
   , int* neighborsWeight_start_offset
   , int* neighborsWeight_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , float** neighborsWeight
   , int neighborsWeight_max_elements
   , int* neighborsWeight_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< float* >* neighborsWeight
   //need more info here
   #endif

   , unsigned size
   , int complexity
   , int actionType
   , int tooCrowded
   , int tooSparse
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
      float weightSum = 0;
      float learnRate = 0.0001;
      float dw = 0;
       // add your code here
      for (int ii = 0; ii < complexity; ii++)
      {
#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
         auto um_neighborsWeight_from = index * neighborsWeight_max_elements;
         auto um_neighborsWeight_to = um_neighborsWeight_from + neighborsWeight_num_elements[index];
         for (auto idx = um_neighborsWeight_from; idx < um_neighborsWeight_to; ++idx) {
            weightSum += *(neighborsWeight[idx]);
         }
#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
#endif
         if (actionType == F_SIGMOID)
            dw += sigmoid(weightSum);
         else if (actionType == F_ReLU)
            dw += ReLU(weightSum);
         else if (actionType == F_TANH)
            dw += tanh(weightSum);

         if (value[index] == 1) {
            weight[index] += dw;
         }
         else {
            weight[index] -= dw;
         }
      }
   }
}
#endif
