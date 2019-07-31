#ifndef SupervisorNode_H
#define SupervisorNode_H

#include "Lens.h"
#include "CG_SupervisorNode.h"
#include "rndm.h"

class SupervisorNode : public CG_SupervisorNode
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual ~SupervisorNode();
};

#endif
