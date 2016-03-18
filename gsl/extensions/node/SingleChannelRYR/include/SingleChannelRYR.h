#ifndef SingleChannelRYR_H
#define SingleChannelRYR_H

#include "Lens.h"
#include "CG_SingleChannelRYR.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

class SingleChannelRYR : public CG_SingleChannelRYR
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~SingleChannelRYR();

	void updateChannelTransitionRate(dyn_var_t* & matChannelTransitionRate, int cptIdx);
	private:
};

#endif
