#ifndef SingleChannelRYR_H
#define SingleChannelRYR_H

#include "Lens.h"
#include "CG_SingleChannelRYR.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#ifndef BASED_TEMPERATURE
#define BASED_TEMPERATURE 35.0  // Celcius
#endif

#ifndef Q10
#define Q10 3.0  // default
#endif
class SingleChannelRYR : public CG_SingleChannelRYR
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~SingleChannelRYR();

  void updateChannelTransitionRate(dyn_var_t*& matChannelTransitionRate,
                                   int cptIdx);

  private:
};

#endif
