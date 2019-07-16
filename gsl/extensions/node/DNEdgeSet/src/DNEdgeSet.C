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
  for (unsigned n=0; n<sz; ++n) {
    weights[n] = drandom(rng);
    weightedOutputs[n] = PRELIM_STATE;
  }
  readyForward = false;
  readyBackward = false;
  transferFunction.setType(SHD.transferFunctionName);
}

void DNEdgeSet::update(RNG& rng) 
{
  if (!readyForward)
    readyForward = *input != PRELIM_STATE;
    
  ShallowArray<double*>::iterator giter, gend = gradients.end();
  if (readyForward) {
    ShallowArray<double>::iterator witer=weights.begin(),
      woiter=weightedOutputs.begin(),
      woend=weightedOutputs.end();

    for (; woiter!=woend; ++woiter, ++witer) {
      *woiter = *witer * transferFunction.transfer(*input);
    }
    if (!readyBackward) {
      for (giter=gradients.begin(); giter!=gend; ++giter) {
	readyBackward = **giter != PRELIM_STATE;
	if (!readyBackward) {
	  echoes.push_back(*input);
	  break;
	}
      }
    }
    if (readyBackward) {
      double dow = 0;
      witer=weights.begin();
      for (giter=gradients.begin(); giter!=gend; ++giter, ++witer) {
	dow +=  *witer * **giter;
	double deltaWeight =
	  (1-SHD.alpha) * SHD.eta * transferFunction.transfer(echoes[echoIndex]) * **giter +
	  SHD.alpha * oldDeltaWeight;
	
	*witer += deltaWeight;
	oldDeltaWeight = deltaWeight;
      }

      weightedGradient = dow * transferFunction.derivativeOfTransfer(echoes[echoIndex]);
      echoes[echoIndex] = *input;
      if (++echoIndex == echoes.size()) echoIndex = 0; 
    }
  }
}

DNEdgeSet::~DNEdgeSet() 
{
}

