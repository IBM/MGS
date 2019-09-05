#include "Lens.h"
#include "DNEdgeSet.h"
#include "CG_DNEdgeSet.h"
#include "rndm.h"
#ifdef HAVE_GPU
#include "CG_DNEdgeSetCompCategory.h"
#endif

#ifdef HAVE_GPU

#define weights  (_container->um_weights[index])
#define deltaWeights  (_container->um_deltaWeights[index])
#define deltaWeightsSquared  (_container->um_deltaWeightsSquared[index])
#define weightedOutputs  (_container->um_weightedOutputs[index])
#define weightedGradient  (_container->um_weightedGradient[index])
#define echoes  (_container->um_echoes[index])
#define echoIndex  (_container->um_echoIndex[index])
#define biasCorrectionW  (_container->um_biasCorrectionW[index])
#define biasCorrectionS  (_container->um_biasCorrectionS[index])
#define input  (_container->um_input[index])
#define gradients  (_container->um_gradients[index])
#define readyForward  (_container->um_readyForward[index])
#define readyBackward  (_container->um_readyBackward[index])
#define transferFunctionName  (_container->um_transferFunctionName[index])
#define momentum  (_container->um_momentum[index])
#define rmsprop  (_container->um_rmsprop[index])
#endif

#define PRELIM_STATE DBL_MAX
#define SMALL_NUMBER 0.00000001
#define SHD getSharedMembers()

void DNEdgeSet::initialize(RNG& rng) 
{
  /*
  if (bias) {
    std::cerr<<*bias<<std::endl;
    input=bias;
  }
  */
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

DNEdgeSet::~DNEdgeSet() 
{
}

