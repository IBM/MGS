#ifndef LifeNode_H
#define LifeNode_H

#include "Lens.h"
#include "CG_LifeNode.h"
#include "rndm.h"

class LifeNode : public CG_LifeNode
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void copy(RNG& rng);
      virtual ~LifeNode();
};

#endif
