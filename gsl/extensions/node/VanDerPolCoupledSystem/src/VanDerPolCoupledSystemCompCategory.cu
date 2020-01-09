#ifndef CG_VANDERPOLCOUPLEDSYSTEMCOMPCATEGORY_CU
#define CG_VANDERPOLCOUPLEDSYSTEMCOMPCATEGORY_CU
void __global__ VanDerPolCoupledSystem_kernel_initializeNode(
   int* m
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x1
   , int* x1_start_offset
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x1
   , int x1_max_elements
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x2
   , int* x2_start_offset
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x2
   , int x2_max_elements
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha1
   , int* alpha1_start_offset
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha1
   , int alpha1_max_elements
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha2
   , int* alpha2_start_offset
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha2
   , int alpha2_max_elements
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* W
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* W
   , int* W_start_offset
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* W
   , int W_max_elements
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* W
   //need more info here
   #endif

   , double* u
   , unsigned size
   , bool saveBinary
   , int collectWeightsNext
   , int* collectWeightsOn
   , String sharedFileExt
   , String sharedDirectory
   , String json_file
   , double predictionFactor
   , double dT
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
       printf("Implement the kernel VanDerPolCoupledSystem_kernel_initializeNode");
       assert(0);
   }
}
void __global__ VanDerPolCoupledSystem_kernel_initializeSolver(
   int* m
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x1
   , int* x1_start_offset
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x1
   , int x1_max_elements
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x2
   , int* x2_start_offset
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x2
   , int x2_max_elements
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha1
   , int* alpha1_start_offset
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha1
   , int alpha1_max_elements
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha2
   , int* alpha2_start_offset
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha2
   , int alpha2_max_elements
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* W
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* W
   , int* W_start_offset
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* W
   , int W_max_elements
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* W
   //need more info here
   #endif

   , double* u
   , unsigned size
   , bool saveBinary
   , int collectWeightsNext
   , int* collectWeightsOn
   , String sharedFileExt
   , String sharedDirectory
   , String json_file
   , double predictionFactor
   , double dT
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
       printf("Implement the kernel VanDerPolCoupledSystem_kernel_initializeSolver");
       assert(0);
   }
}
void __global__ VanDerPolCoupledSystem_kernel_update1(
   int* m
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x1
   , int* x1_start_offset
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x1
   , int x1_max_elements
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x2
   , int* x2_start_offset
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x2
   , int x2_max_elements
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha1
   , int* alpha1_start_offset
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha1
   , int alpha1_max_elements
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha2
   , int* alpha2_start_offset
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha2
   , int alpha2_max_elements
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* W
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* W
   , int* W_start_offset
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* W
   , int W_max_elements
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* W
   //need more info here
   #endif

   , double* u
   , unsigned size
   , bool saveBinary
   , int collectWeightsNext
   , int* collectWeightsOn
   , String sharedFileExt
   , String sharedDirectory
   , String json_file
   , double predictionFactor
   , double dT
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
       printf("Implement the kernel VanDerPolCoupledSystem_kernel_update1");
       assert(0);
   }
}
void __global__ VanDerPolCoupledSystem_kernel_update2(
   int* m
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x1
   , int* x1_start_offset
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x1
   , int x1_max_elements
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x2
   , int* x2_start_offset
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x2
   , int x2_max_elements
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha1
   , int* alpha1_start_offset
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha1
   , int alpha1_max_elements
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha2
   , int* alpha2_start_offset
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha2
   , int alpha2_max_elements
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* W
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* W
   , int* W_start_offset
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* W
   , int W_max_elements
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* W
   //need more info here
   #endif

   , double* u
   , unsigned size
   , bool saveBinary
   , int collectWeightsNext
   , int* collectWeightsOn
   , String sharedFileExt
   , String sharedDirectory
   , String json_file
   , double predictionFactor
   , double dT
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
       printf("Implement the kernel VanDerPolCoupledSystem_kernel_update2");
       assert(0);
   }
}
void __global__ VanDerPolCoupledSystem_kernel_update3(
   int* m
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x1
   , int* x1_start_offset
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x1
   , int x1_max_elements
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x2
   , int* x2_start_offset
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x2
   , int x2_max_elements
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha1
   , int* alpha1_start_offset
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha1
   , int alpha1_max_elements
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha2
   , int* alpha2_start_offset
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha2
   , int alpha2_max_elements
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* W
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* W
   , int* W_start_offset
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* W
   , int W_max_elements
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* W
   //need more info here
   #endif

   , double* u
   , unsigned size
   , bool saveBinary
   , int collectWeightsNext
   , int* collectWeightsOn
   , String sharedFileExt
   , String sharedDirectory
   , String json_file
   , double predictionFactor
   , double dT
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
       printf("Implement the kernel VanDerPolCoupledSystem_kernel_update3");
       assert(0);
   }
}
void __global__ VanDerPolCoupledSystem_kernel_update4(
   int* m
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x1
   , int* x1_start_offset
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x1
   , int x1_max_elements
   , int* x1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* x2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* x2
   , int* x2_start_offset
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* x2
   , int x2_max_elements
   , int* x2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* x2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha1
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha1
   , int* alpha1_start_offset
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha1
   , int alpha1_max_elements
   , int* alpha1_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha1
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* alpha2
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* alpha2
   , int* alpha2_start_offset
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* alpha2
   , int alpha2_max_elements
   , int* alpha2_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* alpha2
   //need more info here
   #endif

   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* W
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* W
   , int* W_start_offset
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* W
   , int W_max_elements
   , int* W_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* W
   //need more info here
   #endif

   , double* u
   , unsigned size
   , bool saveBinary
   , int collectWeightsNext
   , int* collectWeightsOn
   , String sharedFileExt
   , String sharedDirectory
   , String json_file
   , double predictionFactor
   , double dT
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
       printf("Implement the kernel VanDerPolCoupledSystem_kernel_update4");
       assert(0);
   }
}
#endif
