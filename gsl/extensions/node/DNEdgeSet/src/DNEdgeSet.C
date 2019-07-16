#include "Lens.h"
#include "DNEdgeSet.h"
#include "CG_DNEdgeSet.h"
#include "rndm.h"

#define PRELIM_STATE DBL_MAX
#define SMALL_NUMBER 0.00000001
#define SHD getSharedMembers()

void DNEdgeSet::initialize(RNG& rng) 
{
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
  transferFunction.setType(transferFunctionName);
  biasCorrectionW = SHD.alpha;
  biasCorrectionS = SHD.beta;

  ShallowArray<String>::const_iterator oiter, oend = SHD.optimization.end();
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
    
  ShallowArray<double*>::iterator giter, gend = gradients.end();
  if (readyForward) {
    ShallowArray<double>::iterator witer=weights.begin(),
      diter=deltaWeights.begin(),
      siter=deltaWeightsSquared.begin(),
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
      assert(getSimulation().getIteration()>0);
      
      witer=weights.begin();      
      for (giter=gradients.begin(); giter!=gend; ++giter, ++witer, ++diter, ++siter) {
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

