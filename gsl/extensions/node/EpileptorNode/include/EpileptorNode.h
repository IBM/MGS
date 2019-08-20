#ifndef EpileptorNode_H
#define EpileptorNode_H

#include "Lens.h"
#include "CG_EpileptorNode.h"
#include "rndm.h"

class EpileptorNode : public CG_EpileptorNode
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void copy(RNG& rng);
      virtual ~EpileptorNode();
};

#endif
