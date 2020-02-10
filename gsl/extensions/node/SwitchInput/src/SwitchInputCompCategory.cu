#ifndef CG_SWITCHINPUTCOMPCATEGORY_CU
#define CG_SWITCHINPUTCOMPCATEGORY_CU
void __global__ SwitchInput_kernel_initialize(
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* drivinps
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   , double* drivinps
   , int* drivinps_start_offset
   , int* drivinps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   , double* drivinps
   , int drivinps_max_elements
   , int* drivinps_num_elements
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   , ShallowArray< double >* drivinps
   //need more info here
   #endif

   , double* drivinp
   , double* inplo
   , double* inphi
   , unsigned size
   , double period
   , double refract
   , unsigned seqlen
   , unsigned inpnum
   , unsigned statenum
   , unsigned currentstate
   , double* stateswitchtimes
   , unsigned* stateseq
   , double var2
   , double var1
   , double tscale
   , double noiselev
   , double deltaT
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
       printf("Implement the kernel SwitchInput_kernel_initialize");
       assert(0);
   }
}
void __global__ SwitchInput_kernel_update(
   double* drivinps
   , double* drivinp
   , unsigned size
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
      // add your code here
      //drivinp[index] = drivinps[index + inpnum*stride];
      drivinp[index] = drivinps[index];
   }
}
#endif
