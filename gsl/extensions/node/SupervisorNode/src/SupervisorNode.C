#include "Lens.h"
#include "SupervisorNode.h"
#include "CG_SupervisorNode.h"
#include "rndm.h"

#define PRELIM_STATE DBL_MAX
#define SHD getSharedMembers()

void SupervisorNode::initialize(RNG& rng) 
{
  transferFunction.setType(SHD.transferFunctionName);
}

void SupervisorNode::update(RNG& rng) 
{
  double oneHot = (SHD.label == getGlobalIndex()) ? 1.0 : 0;
  double error = oneHot - *prediction;
  if (SHD.refreshErrors) sumOfSquaredError=0;
  sumOfSquaredError += error * error;
  primaryGradient = (oneHot - *prediction) * transferFunction.derivativeOfTransfer(*prediction);
}
SupervisorNode::~SupervisorNode() 
{
}

