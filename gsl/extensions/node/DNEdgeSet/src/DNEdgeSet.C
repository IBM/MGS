// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "DNEdgeSet.h"
#include "CG_DNEdgeSet.h"
#include "rndm.h"
#include <cfloat>
#ifdef HAVE_GPU
#include "CG_DNEdgeSetCompCategory.h"
#endif

#ifdef HAVE_GPU

#define weights  (_container->um_weights[__index__])
#define deltaWeights  (_container->um_deltaWeights[__index__])
#define deltaWeightsSquared  (_container->um_deltaWeightsSquared[__index__])
#define weightedOutputs  (_container->um_weightedOutputs[__index__])
#define weightedGradient  (_container->um_weightedGradient[__index__])
#define echoes  (_container->um_echoes[__index__])
#define echoIndex  (_container->um_echoIndex[__index__])
#define biasCorrectionW  (_container->um_biasCorrectionW[__index__])
#define biasCorrectionS  (_container->um_biasCorrectionS[__index__])
#define input  (_container->um_input[__index__])
#define gradients  (_container->um_gradients[__index__])
#define readyForward  (_container->um_readyForward[__index__])
#define readyBackward  (_container->um_readyBackward[__index__])
#define transferFunctionName  (_container->um_transferFunctionName[__index__])
#define momentum  (_container->um_momentum[__index__])
#define rmsprop  (_container->um_rmsprop[__index__])

#define udef_fncIndex (_container->udef_um_fncIndex[__index__])
#endif

#define PRELIM_STATE DBL_MAX
#define SMALL_NUMBER 0.00000001
#define SHD getSharedMembers()

void DNEdgeSet::initialize(RNG& rng) 
{
#if defined(HAVE_GPU)
  // must be pre-allocated to work on GPU
  echoes.resize_allocated(SHD.max_num_layers);
  // track the index to the transfer function
  udef_fncIndex = transferFunction.setType(transferFunctionName);
  //...
#else
  transferFunction.setType(transferFunctionName);
#endif
  unsigned sz = gradients.size();
  weightedOutputs.increaseSizeTo(sz);
  weights.increaseSizeTo(sz);
  deltaWeights.increaseSizeTo(sz);
  deltaWeightsSquared.increaseSizeTo(sz);
  for (unsigned n=0; n<sz; ++n) {
    weights[n] = drandom(rng) * 2 - 1.0;
    weightedOutputs[n] = PRELIM_STATE;
    deltaWeights[n] = 0.0;
    deltaWeightsSquared[n] = 0.0;
  }
  weightedGradient = PRELIM_STATE;
  readyForward = false;
  readyBackward = false;

  biasCorrectionW = SHD.alpha;
  biasCorrectionS = SHD.beta;

  ShallowArray<CustomString>::const_iterator oiter, oend = SHD.optimization.end();
  for (oiter=SHD.optimization.begin(); oiter!=oend; ++oiter) {
    if (*oiter=="Momentum") momentum=true;
    else if (*oiter=="RMSprop") rmsprop=true;
    else if (*oiter=="Adam") momentum=rmsprop=true;
  }  
}

void DNEdgeSet::update(RNG& rng) 
{
  if (!readyForward)
    readyForward = *input != PRELIM_STATE;
    
  auto gend = gradients.end();
  if (readyForward) {
    auto witer=weights.begin();
    auto diter=deltaWeights.begin();
    auto siter=deltaWeightsSquared.begin();
    auto woiter=weightedOutputs.begin();
    auto woend=weightedOutputs.end();

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

DNEdgeSet::~DNEdgeSet() 
{
}

