#ifndef LIFENODECOMPCATEGORY_CU
#define LIFENODECOMPCATEGORY_CU
void __global__ LifeNode_kernel_initialize(
     int* value,
     int* publicValue,
     ShallowArray_Flat<int*, Array_Flat<int>::MemLocation::UNIFIED_MEM>* neighbors,  
     //RNG& rng
     unsigned size
     , int tooSparse 
     , int tooCrowded
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
     ShallowArray_Flat<int*, Array_Flat<int>::MemLocation::UNIFIED_MEM>* neighbors,  
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
      ShallowArray_Flat<int*>::iterator end = neighbors[index].end();
      for (auto iter=neighbors[index].begin(); iter!=end; ++iter) {
         neighborCount += **iter;
      }

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
     ShallowArray_Flat<int*, Array_Flat<int>::MemLocation::UNIFIED_MEM>* neighbors,  
     //RNG& rng
     unsigned size
     , int tooSparse 
     , int tooCrowded
     ) 
{
   int index =  blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size)
   {
      publicValue[index]=value[index];
   }
}
#endif
