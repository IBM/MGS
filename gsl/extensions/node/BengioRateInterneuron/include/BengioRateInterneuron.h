#ifndef BengioRateInterneuron_H
#define BengioRateInterneuron_H

#include "Lens.h"
#include "CG_BengioRateInterneuron.h"
#include "rndm.h"

class BengioRateInterneuron : public CG_BengioRateInterneuron
{
   public:
      void update_U(RNG& rng);
      void update_V(RNG& rng);
      virtual ~BengioRateInterneuron();
};

#endif
