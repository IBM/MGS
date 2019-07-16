#include "Lens.h"
#include "SupervisorNode.h"
#include "CG_SupervisorNode.h"
#include "rndm.h"
#include "IsToast.h"
#include <math.h>

#define PRELIM_STATE DBL_MAX
#define SHD getSharedMembers()

void SupervisorNode::initialize(RNG& rng) 
{
  primaryGradient = PRELIM_STATE;
  logits.increaseSizeTo(predictions.size());
}

void SupervisorNode::update(RNG& rng) 
{
  ShallowArray<double*>::iterator iter, end = predictions.end();  
  if (!ready) {
    for (iter=predictions.begin(); iter!=end; ++iter) {
      ready = (**iter != PRELIM_STATE);
      if (!ready) break;
    }
  }
  if (ready) {
    double sumOfExp=0;
    for (iter=predictions.begin(); iter!=end; ++iter) 
      sumOfExp+=exp(**iter);

    double oneHot = (SHD.labels[SHD.labelIndex] == getGlobalIndex()) ? 1.0 : 0.0;

    ShallowArray<double>::iterator my_liter, liter=logits.begin(), lend=logits.end();

    unsigned h=0;
    unsigned winner=-1;
    double maxLogit=-DBL_MAX;
    for (iter=predictions.begin(); iter!=end; ++iter, ++liter, ++h) {
      *liter = exp(**iter)/sumOfExp;
      if (*liter>maxLogit) {
	winner=h;
	maxLogit=*liter;
      }
    }

    if (winner==SHD.labels[SHD.labelIndex]) ++wins;
    
    my_liter = logits.begin()+getGlobalIndex();

    double error = oneHot - *my_liter;
    if (SHD.refreshErrors) {
      sumOfSquaredError=0;
      wins=0;
    }
    sumOfSquaredError += error * error;

    primaryGradient=0;
    if (!SHD.test) {
      h=0;
      for (liter=logits.begin(), iter=predictions.begin(); liter!=lend; ++h, ++liter, ++iter) {
	primaryGradient += *my_liter * ( ( (liter==my_liter) ? 1.0 : 0.0 ) - *liter ) *
	  ( ( (SHD.labels[SHD.labelIndex] == h) ? 1.0 : 0.0 ) - *liter );
      }
    }
  }
}

SupervisorNode::~SupervisorNode() 
{
}

