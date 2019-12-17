#ifndef CG_MAHONUNITCOMPCATEGORY_CU
#define CG_MAHONUNITCOMPCATEGORY_CU

#define q10 2.5
#define CELSIUS 37
#define ETEMPKAF 22 //KAf, KAS, Krpm, Nap, 22
#define ETEMPNAS 21 
#define TADJPKAF 3.953
#define TADJPNAS 4.332


#define GLEAK 0.075
#define VLEAK -90.0 //-75.0 //(Mahon) //-90.0 (Gittis)
#define GNAS 0.11
#define VNAS 40.0
#define GNAP 0.02
#define VNAP 45.0
#define GKRP 0.42
#define VKRP -77.5
#define GAS 0.32
#define VAS -85.0
#define GAF 0.09
#define VAF -73.0
#define GKIR 0.15
#define VKIR -90.0
#define GKCHAN 6.0
#define VKCHAN -90.0
#define GNACHAN 35.0
#define VNACHAN 55.0

#define NASMTHE -16.0 
#define NASMK 9.4 
#define NASMTAU1 637.8
#define NASMPHI -33.5
#define NASMSIG0 26.3

#define NAPMTHE -47.8 
#define NAPMK 3.1 
#define NAPMTAU1 1.0

#define KRPMTHE -13.4 
#define KRPMK 12.1 
#define KRPMTAU1 206.2 
#define KRPMPHI -53.9 
#define KRPMSIG0 26.5

#define KRPHTHE -55.0 
#define KRPHK -19.0 

#define ASHTHE -78.8
#define ASHK -10.4

#define XASMTHE -25.6
#define XASMK 13.3
#define XASMTAU1 131.4
#define XASMPHI -37.4
#define XASMSIG0 27.3

#define AFMTHE -33.1 
#define AFMK 7.5
#define AFMTAU1 1.0

#define AFHTHE -70.4 
#define AFHK -7.6
#define AFHTAU1 25.0

#define KIRMTHE -100.0
#define KIRMK -10.0
#define KIRMTAU1 0.01

#define VCL -80.0
#define SYNA 2
#define SYNB 0.1

#define CAPAC 1.0


#define NUM_VARS_PER_NODE 12

__device__ __forceinline__ int MahonUnit_find_var_index(int logical_index, 
   int var_index, int size)
{
    return logical_index + var_index * size;
}


void __global__ MahonUnit_kernel_initialize(
   double* V_init
   , double* x
   , unsigned size /* # of nodes */
   , unsigned stride  /* stride length >= size */
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
      //, double* x, //time-dependent variables
      //double & V    = x[(index+0*stride)];
      //V=V_init[index];

      double V=V_init[index];
      x[(index+0*stride)] = V;

      double & Nasm = x[(index+1*stride)];
      Nasm = sigmoid<double>(V,NASMTHE,NASMK);
      double & Napm = x[(index+2*stride)];
      Napm = sigmoid<double>(V,NAPMTHE,NAPMK);
      double & Krpm = x[(index+3*stride)];
      Krpm = sigmoid<double>(V,KRPMTHE,KRPMK);
      double & Krph = x[(index+4*stride)];
      Krph = sigmoid<double>(V,KRPHTHE,KRPHK);
      double & Asm  = x[(index+5*stride)];
      Asm = sigmoid<double>(V,XASMTHE,XASMK);
      double & Ash  = x[(index+6*stride)];
      Ash = sigmoid<double>(V,ASHTHE,ASHK);
      double & Afm  = x[(index+7*stride)];
      Afm = sigmoid<double>(V,AFMTHE,AFMK);
      double & Afh  = x[(index+8*stride)];
      Afh = sigmoid<double>(V,AFHTHE,AFHK);
      double & Km   = x[(index+9*stride)];  
      Km = gatefcnInstant<double>(Kmalpha<double>(V),Kmbeta<double>(V));
      double & Nah  = x[(index+10*stride)];
      Nah = gatefcnInstant<double>(Nahalpha<double>(V),Nahbeta<double>(V));
      //double & Kirm = x[(index+11*stride)];
      //Kirm = sigmoid<float>(V,KIRMTHE,KIRMK);
   }else if (index < stride)
   {
      for (int ii=0; ii < NUM_VARS_PER_NODE; ii++)
         x[(index+ii*stride)] = 0.0f;
   }
}

void __global__ MahonNet_derivs(
//const ShallowArray< double > & x, ShallowArray< double > & dx)
  double* x
  , double* dx
  , ShallowArray_Flat<Input, Array_Flat<int>::MemLocation::UNIFIED_MEM>* MSNNetInps
   , double** drivinp
   , double* synb
   , size_t size
   , size_t stride
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
      //ShallowArray<Input>::iterator iter, end=MSNNetInps.end();
      auto end=MSNNetInps[index].end();
      double drive=0;
      for (auto iter=MSNNetInps[index].begin(); iter!=end; ++iter) {
         drive += *(iter->input)*iter->weight;
      }
      
      const double & V    = x[(index+0*stride)];
      const double & Nasm = x[(index+1*stride)];
      const double & Napm = x[(index+2*stride)];
      const double & Krpm = x[(index+3*stride)];
      const double & Krph = x[(index+4*stride)];
      const double & Asm  = x[(index+5*stride)];
      const double & Ash  = x[(index+6*stride)];
      const double & Afm  = x[(index+7*stride)];
      const double & Afh  = x[(index+8*stride)];
      const double & Km   = x[(index+9*stride)];
      const double & Nah  = x[(index+10*stride)];
      const double & g    = x[(index+11*stride)];

      double & dV    = dx[(index+0*stride)];
      double & dNasm = dx[(index+1*stride)];
      double & dNapm = dx[(index+2*stride)];
      double & dKrpm = dx[(index+3*stride)];
      double & dKrph = dx[(index+4*stride)];
      double & dAsm  = dx[(index+5*stride)];
      double & dAsh  = dx[(index+6*stride)];
      double & dAfm  = dx[(index+7*stride)];
      double & dAfh  = dx[(index+8*stride)];
      double & dKm   = dx[(index+9*stride)];
      double & dNah  = dx[(index+10*stride)];
      double & dg    = dx[(index+11*stride)];

      {//dV
         dV = drive*(V - VCL);
         //std::cout << dV << std::endl; 
         //std::ofstream syn_curr_file;
         //syn_curr_file.open("syn_curr.txt",std::ofstream::out | std::ofstream::app);
         //syn_curr_file << dV << " "; //std::endl;
         //syn_curr_file.close();
         //dV = 0;

         dV -= IonChannel<double>(V,GLEAK,VLEAK);

         dV -= IonChannel<double>(V,Nasm,GNAS,VNAS);
         dV -= IonChannel<double>(V,Napm,GNAP,VNAP);
         dV -= IonChannel<double>(V,Krpm,Krph,GKRP,VKRP);
         dV -= IonChannel<double>(V,Asm,Ash,GAS,VAS);
         dV -= IonChannel<double>(V,Afm,Afh,GAF,VAF);

         //dV -= IonChannel<double>(V,Kirm,GKIR,VKIR);

         dV -= IonChannel<double>(V,sigmoid<float>(V,KIRMTHE,KIRMK),GKIR,VKIR);

         dV -= IonChannel4<double>(V,Km,GKCHAN,VKCHAN);
         dV -= IonChannel31<double>
            (V,gatefcnInstant<double>(Namalpha<double>(V),Nambeta<double>(V)),Nah,GNACHAN,VNACHAN);

         //const int t2 = TIME;
         //const int t1 = t2 % 400;

         //if (t2>10000 && t1>200) dV += injCur;
         //if (t2>10000) dV += injCur;

         dV -= (*drivinp[index])*V;
         dV /=CAPAC;
      }
      const double val1 = Ashtaufcn(V);

      {//dNasm
      dNasm = ratefcn<double>(Nasm,sigmoid<float>(V,NASMTHE,NASMK),
                              TadjAdj<double>(taufcn<float>(V,NASMTAU1,NASMPHI,NASMSIG0),TADJPNAS));
      }

      {//dNapm
      dNapm = ratefcn<double>(Napm,sigmoid<float>(V,NAPMTHE,NAPMK),TadjAdj<double>(NAPMTAU1,TADJPKAF));

      }

      {//dKrpm
      dKrpm = ratefcn<double>(Krpm,sigmoid<float>(V,KRPMTHE,KRPMK),
                              TadjAdj<double>(taufcn<float>(V,KRPMTAU1,KRPMPHI,KRPMSIG0),TADJPKAF));
      }

      {//dKrph
      dKrph = ratefcn<double>(Krph,sigmoid<float>(V,KRPHTHE,KRPHK),TadjAdj<double>(val1*3.0,TADJPKAF));
      }

      {//dAsm
      dAsm = ratefcn<double>(Asm,sigmoid<float>(V,XASMTHE,XASMK),
                              TadjAdj<double>(taufcn<float>(V,XASMTAU1,XASMPHI,XASMSIG0),TADJPKAF));
      }

      {//dAsh
      dAsh = ratefcn<double>(Ash,sigmoid<float>(V,ASHTHE,ASHK),TadjAdj<double>(val1,TADJPKAF));
      }

      {//dAfm
      dAfm = ratefcn<double>(Afm,sigmoid<float>(V,AFMTHE,AFMK),TadjAdj<double>(AFMTAU1,TADJPKAF));
      }

      {//dAfh
      dAfh = ratefcn<double>(Afh,sigmoid<float>(V,AFHTHE,AFHK),TadjAdj<double>(AFHTAU1,TADJPKAF));
      }

      {//dKm
      dKm = gatefcn<double>(Km,Kmalpha<double>(V),Kmbeta<double>(V),5.0);
      }

      {//dNah
      dNah = gatefcn<double>(Nah,Nahalpha<double>(V),Nahbeta<double>(V),5.0);
      }

      //const double & Nam = x[11];
      //double & dNam = dx[11];
      //  const double & Kirm = x[11];
      //double & dKirm = dx[11];
      //dKirm = ratefcn<double>(Kirm,sigmoid<float>(V,KIRMTHE,KIRMK),TadjAdj<double>(KIRMTAU1,TADJPKAF));

      dg = SYNA*((double) (V>0.0))*(1.0 - g) - synb[index]*g;
  }
}

void __global__ MahonUnit_kernel_flushVars1(
   double* g_out
   , double* x
   , unsigned size
   , unsigned stride  /* stride length >= size */
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
      g_out[index] = x[index+11*stride];
   }
}
void __global__ MahonUnit_kernel_flushVars2(
   double* g_out
   , double* x
   , unsigned size
   , unsigned stride  /* stride length >= size */
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
      g_out[index] = x[index+11*stride];
   }
}
void __global__ MahonUnit_kernel_flushVars3(
   double* g_out
   , double* x
   , unsigned size
   , unsigned stride  /* stride length >= size */
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
      g_out[index] = x[index+11*stride];
   }
}
void __global__ MahonUnit_kernel_flushVars4(
   double* g_out
   , double* x
   , unsigned size
   , unsigned stride  /* stride length >= size */
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
      g_out[index] = x[index+11*stride];
   }
}
void __global__ MahonUnit_kernel_updateOutputs(
   double* var1
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
   , bool* spike
   , double spikethresh
   , double* x
   , unsigned size
   , unsigned stride  /* stride length >= size */
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
      double V = x[index + 0*stride];
      if (V >= spikethresh && var1[index] < spikethresh) 
         spike[index] = true;
      else 
         spike[index] = false;
      var1[index] = V;
      //var2 = x[5];

      auto end=MSNNetInps[index].end();
      double drive=0;
      for (auto iter=MSNNetInps[index].begin(); iter!=end; ++iter) {
         drive += *(iter->input)*iter->weight;
      }
      var3[index] = drive; //x[11];
      var2[index] = drive*(V - VCL);
   }
}
#endif
