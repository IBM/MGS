#ifndef CG_DNEDGESETCOMPCATEGORY_CU
#define CG_DNEDGESETCOMPCATEGORY_CU
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
   , String* transferFunctionName
   , bool* momentum
   , bool* rmsprop
   , unsigned size
   , ShallowArray< String > optimization
   , double beta
   , double alpha
   , double eta
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
       // add your code here
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
   , String* transferFunctionName
   , bool* momentum
   , bool* rmsprop
   , unsigned size
   , ShallowArray< String > optimization
   , double beta
   , double alpha
   , double eta
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
      if (!readyForward)
          readyForward = *(input[index]) != PRELIM_STATE;
   auto gend = gradients.end();
   if (readyForward) {
      auto witer=weights.begin(),
          diter=deltaWeights.begin(),
      	  siter=deltaWeightsSquared.begin(),
      	  woiter=weightedOutputs.begin(),
      	  woend=weightedOutputs.end();

      double transferInput = transferFunction.transfer(*input);
      
      for (; woiter!=woend; ++woiter, ++witer) {
      *woiter = *witer * transferInput;
    }
    if (!readyBackward) {
      for (auto giter=gradients.begin(); giter!=gend; ++giter) {
	readyBackward = **giter != PRELIM_STATE;
	if (!readyBackward) {	  
	  echoes.push_back(transferInput);	  
	  break;
	}
      }
    }
    if (readyBackward) {
      double dow = 0;
      assert(getSimulation().getIteration()>0);
      
      witer=weights.begin();      
      for (auto giter=gradients.begin(); giter!=gend; ++giter, ++witer, ++diter, ++siter) {
	dow +=  *witer * **giter;

	double deltaWeight = echoes[echoIndex] * **giter;
	double update = SHD.eta;

	if (rmsprop)
	  *siter = ( (1-SHD.beta) * deltaWeight * deltaWeight + SHD.beta * *siter ) / (1.0 - biasCorrectionS);

	if (momentum) {
	  *diter = ( (1-SHD.alpha) * deltaWeight + SHD.alpha * *diter ) / (1.0 - biasCorrectionW);
	  update *=  *diter;
	}
	else
	  update *= deltaWeight;
	
	if (rmsprop)
	  update /= sqrt(*siter + SMALL_NUMBER);

	*witer += update;
      }

      if (momentum) biasCorrectionW *= SHD.alpha;
      if (rmsprop) biasCorrectionS *= SHD.beta;

      weightedGradient = dow * transferFunction.derivativeOfTransfer(echoes[echoIndex]);
      echoes[echoIndex] = transferInput;
      if (++echoIndex == echoes.size()) echoIndex = 0; 
    }
  }
	  
   }
}
#endif
