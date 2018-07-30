#ifndef ECS_Medium_H
#define ECS_Medium_H

#include "Lens.h"
#include "CG_ECS_Medium.h"
#include "rndm.h"

class ECS_Medium : public CG_ECS_Medium
{
   public:
      void initParams(RNG& rng);
      void update(RNG& rng);
      void copy(RNG& rng);
      void finalize(RNG& rng);
      virtual ~ECS_Medium();
};

#endif
