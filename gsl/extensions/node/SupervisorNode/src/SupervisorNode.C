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
    double oneHot = (SHD.label == getGlobalIndex()) ? 1.0 : -1.0;

    double tpred = transferFunction.transfer(*prediction);
    double error = oneHot - tpred;
    if (SHD.refreshErrors) {
      sumOfSquaredError=0;
    }
    sumOfSquaredError += error * error;

    primaryGradient = error * transferFunction.derivativeOfTransfer(tpred);
  }
}
SupervisorNode::~SupervisorNode() 
{
}

