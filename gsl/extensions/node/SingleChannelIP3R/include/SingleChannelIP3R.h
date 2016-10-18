#ifndef SingleChannelIP3R_H
#define SingleChannelIP3R_H

#include "Lens.h"
#include "CG_SingleChannelIP3R.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#ifndef BASED_TEMPERATURE
#define BASED_TEMPERATURE 35.0  // Celcius
#endif

#ifndef Q10
#define Q10 3.0  // default
#endif

class SingleChannelIP3R : public CG_SingleChannelIP3R
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~SingleChannelIP3R();

  //user-defined
  void updateChannelTransitionRate(dyn_var_t*& matChannelTransitionRate,
                                   int cptIdx);
};

#endif
