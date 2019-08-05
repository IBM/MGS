#include "Lens.h"
#include "DNEdgeSet.h"
#include "CG_DNEdgeSet.h"
#include "rndm.h"

#define PRELIM_STATE DBL_MAX
#define SHD getSharedMembers()

void DNEdgeSet::initialize(RNG& rng) 
{
  unsigned sz = gradients.size();
  weightedOutputs.increaseSizeTo(sz);
  weights.increaseSizeTo(sz);
  deltaWeights.increaseSizeTo(sz);
  for (unsigned n=0; n<sz; ++n) {
    weights[n] = drandom(rng) * 2 - 1.0;
    weightedOutputs[n] = PRELIM_STATE;
    deltaWeights[n] = 0.0;
  }
  weightedGradient = PRELIM_STATE;
  readyForward = false;
  readyBackward = false;
  transferFunction.setType(transferFunctionName);
}

void DNEdgeSet::update(RNG& rng) 
{
  if (!readyForward)
    readyForward = *input != PRELIM_STATE;
    
  ShallowArray<double*>::iterator giter, gend = gradients.end();
  if (readyForward) {
    ShallowArray<double>::iterator witer=weights.begin(),
      diter=deltaWeights.begin(),
      woiter=weightedOutputs.begin(),
      woend=weightedOutputs.end();

    double transferInput = transferFunction.transfer(*input);
    
    for (; woiter!=woend; ++woiter, ++witer) {
      *woiter = *witer * transferInput;
    }
    if (!readyBackward) {
      for (giter=gradients.begin(); giter!=gend; ++giter) {
	readyBackward = **giter != PRELIM_STATE;
	if (!readyBackward) {	  
	  echoes.push_back(transferInput);	  
	  break;
	}
      }
    }
    if (readyBackward) {
      double dow = 0;
      
      witer=weights.begin();      
      for (giter=gradients.begin(); giter!=gend; ++giter, ++witer, ++diter) {
	dow +=  *witer * **giter;
	double deltaWeight =
	  (1-SHD.alpha) * SHD.eta * echoes[echoIndex] * **giter +
	  SHD.alpha * *diter;
	*witer += deltaWeight;
	*diter = deltaWeight;
      }

      weightedGradient = dow * transferFunction.derivativeOfTransfer(echoes[echoIndex]);
      echoes[echoIndex] = transferInput;
      if (++echoIndex == echoes.size()) echoIndex = 0; 
    }
  }
}

DNEdgeSet::~DNEdgeSet() 
{
}

