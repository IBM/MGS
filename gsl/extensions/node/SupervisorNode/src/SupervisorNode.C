#include "Lens.h"
#include "SupervisorNode.h"
#include "CG_SupervisorNode.h"
#include "rndm.h"
#include "IsToast.h"

#define PRELIM_STATE DBL_MAX
#define SHD getSharedMembers()

void SupervisorNode::initialize(RNG& rng) 
{
  transferFunction.setType(SHD.transferFunctionName);
  primaryGradient = PRELIM_STATE;
}

void SupervisorNode::update(RNG& rng) 
{
  assert(prediction);
  if (!ready) {
    ready = (*prediction != PRELIM_STATE); 
  }
  if (ready) {
    double oneHot = (SHD.label == getGlobalIndex()) ? 1.0 : 0;

    double error = oneHot - transferFunction.transfer(*prediction);
    if (SHD.refreshErrors) {
      sumOfSquaredError=0;
    }
    sumOfSquaredError += error * error;

    primaryGradient = (oneHot - *prediction) * transferFunction.derivativeOfTransfer(*prediction);
  }
}
SupervisorNode::~SupervisorNode() 
{
}

