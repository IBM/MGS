#ifndef ChannelIP3R_H
#define ChannelIP3R_H

#include "Lens.h"
#include "CG_ChannelIP3R.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#ifndef BASED_TEMPERATURE
#define BASED_TEMPERATURE 35.0  // Celcius
#endif

#ifndef Q10
#define Q10 2.3  // default
#endif
class ChannelIP3R : public CG_ChannelIP3R
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelIP3R();
   private:
      dyn_var_t vtrap(dyn_var_t x, dyn_var_t y);
};

#endif
