#ifndef CG_MAHONUNITCOMPCATEGORY_CU
#define CG_MAHONUNITCOMPCATEGORY_CU
void __global__ MahonUnit_kernel_initialize(
   double* g_out
   , double* var1
   , double* var2
   , double* var3
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<Input, Array_Flat<int>::MemLocation::UNIFIED_MEM>* MSNNetInps
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , Input* MSNNetInps
   , int* MSNNetInps_start_offset
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , Input* MSNNetInps
   , int MSNNetInps_max_elements
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< Input >* MSNNetInps
   //need more info here
   #endif

   , double* V_init
   , double* g_init
   , double** drivinp
   , bool* spike
   , double* injCur
   , long* connectionSeed
   , double* synb
   , unsigned size
   , double deltaT
   , double spikethresh
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
   }
}
void __global__ MahonUnit_kernel_update1(
   double* g_out
   , double* var1
   , double* var2
   , double* var3
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<Input, Array_Flat<int>::MemLocation::UNIFIED_MEM>* MSNNetInps
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , Input* MSNNetInps
   , int* MSNNetInps_start_offset
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , Input* MSNNetInps
   , int MSNNetInps_max_elements
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< Input >* MSNNetInps
   //need more info here
   #endif

   , double* V_init
   , double* g_init
   , double** drivinp
   , bool* spike
   , double* injCur
   , long* connectionSeed
   , double* synb
   , unsigned size
   , double deltaT
   , double spikethresh
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
   }
}
void __global__ MahonUnit_kernel_update2(
   double* g_out
   , double* var1
   , double* var2
   , double* var3
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<Input, Array_Flat<int>::MemLocation::UNIFIED_MEM>* MSNNetInps
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , Input* MSNNetInps
   , int* MSNNetInps_start_offset
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , Input* MSNNetInps
   , int MSNNetInps_max_elements
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< Input >* MSNNetInps
   //need more info here
   #endif

   , double* V_init
   , double* g_init
   , double** drivinp
   , bool* spike
   , double* injCur
   , long* connectionSeed
   , double* synb
   , unsigned size
   , double deltaT
   , double spikethresh
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
   }
}
void __global__ MahonUnit_kernel_update3(
   double* g_out
   , double* var1
   , double* var2
   , double* var3
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<Input, Array_Flat<int>::MemLocation::UNIFIED_MEM>* MSNNetInps
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , Input* MSNNetInps
   , int* MSNNetInps_start_offset
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , Input* MSNNetInps
   , int MSNNetInps_max_elements
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< Input >* MSNNetInps
   //need more info here
   #endif

   , double* V_init
   , double* g_init
   , double** drivinp
   , bool* spike
   , double* injCur
   , long* connectionSeed
   , double* synb
   , unsigned size
   , double deltaT
   , double spikethresh
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
   }
}
void __global__ MahonUnit_kernel_update4(
   double* g_out
   , double* var1
   , double* var2
   , double* var3
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<Input, Array_Flat<int>::MemLocation::UNIFIED_MEM>* MSNNetInps
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , Input* MSNNetInps
   , int* MSNNetInps_start_offset
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , Input* MSNNetInps
   , int MSNNetInps_max_elements
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< Input >* MSNNetInps
   //need more info here
   #endif

   , double* V_init
   , double* g_init
   , double** drivinp
   , bool* spike
   , double* injCur
   , long* connectionSeed
   , double* synb
   , unsigned size
   , double deltaT
   , double spikethresh
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
   }
}
void __global__ MahonUnit_kernel_flushVars1(
   double* g_out
   , double* var1
   , double* var2
   , double* var3
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<Input, Array_Flat<int>::MemLocation::UNIFIED_MEM>* MSNNetInps
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , Input* MSNNetInps
   , int* MSNNetInps_start_offset
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , Input* MSNNetInps
   , int MSNNetInps_max_elements
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< Input >* MSNNetInps
   //need more info here
   #endif

   , double* V_init
   , double* g_init
   , double** drivinp
   , bool* spike
   , double* injCur
   , long* connectionSeed
   , double* synb
   , unsigned size
   , double deltaT
   , double spikethresh
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
   }
}
void __global__ MahonUnit_kernel_flushVars2(
   double* g_out
   , double* var1
   , double* var2
   , double* var3
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<Input, Array_Flat<int>::MemLocation::UNIFIED_MEM>* MSNNetInps
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , Input* MSNNetInps
   , int* MSNNetInps_start_offset
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , Input* MSNNetInps
   , int MSNNetInps_max_elements
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< Input >* MSNNetInps
   //need more info here
   #endif

   , double* V_init
   , double* g_init
   , double** drivinp
   , bool* spike
   , double* injCur
   , long* connectionSeed
   , double* synb
   , unsigned size
   , double deltaT
   , double spikethresh
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
   }
}
void __global__ MahonUnit_kernel_flushVars3(
   double* g_out
   , double* var1
   , double* var2
   , double* var3
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<Input, Array_Flat<int>::MemLocation::UNIFIED_MEM>* MSNNetInps
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , Input* MSNNetInps
   , int* MSNNetInps_start_offset
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , Input* MSNNetInps
   , int MSNNetInps_max_elements
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< Input >* MSNNetInps
   //need more info here
   #endif

   , double* V_init
   , double* g_init
   , double** drivinp
   , bool* spike
   , double* injCur
   , long* connectionSeed
   , double* synb
   , unsigned size
   , double deltaT
   , double spikethresh
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
   }
}
void __global__ MahonUnit_kernel_flushVars4(
   double* g_out
   , double* var1
   , double* var2
   , double* var3
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<Input, Array_Flat<int>::MemLocation::UNIFIED_MEM>* MSNNetInps
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , Input* MSNNetInps
   , int* MSNNetInps_start_offset
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , Input* MSNNetInps
   , int MSNNetInps_max_elements
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< Input >* MSNNetInps
   //need more info here
   #endif

   , double* V_init
   , double* g_init
   , double** drivinp
   , bool* spike
   , double* injCur
   , long* connectionSeed
   , double* synb
   , unsigned size
   , double deltaT
   , double spikethresh
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
   }
}
void __global__ MahonUnit_kernel_updateOutputs(
   double* g_out
   , double* var1
   , double* var2
   , double* var3
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<Input, Array_Flat<int>::MemLocation::UNIFIED_MEM>* MSNNetInps
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , Input* MSNNetInps
   , int* MSNNetInps_start_offset
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , Input* MSNNetInps
   , int MSNNetInps_max_elements
   , int* MSNNetInps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< Input >* MSNNetInps
   //need more info here
   #endif

   , double* V_init
   , double* g_init
   , double** drivinp
   , bool* spike
   , double* injCur
   , long* connectionSeed
   , double* synb
   , unsigned size
   , double deltaT
   , double spikethresh
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
   }
}
#endif
