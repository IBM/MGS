#ifndef DNEdgeSet_H
#define DNEdgeSet_H

#include "Lens.h"
#include "CG_DNEdgeSet.h"
#include "rndm.h"
#include "TransferFunction.h"

class DNEdgeSet : public CG_DNEdgeSet
{
   public:
      using Node::initialize;
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual ~DNEdgeSet();
      TransferFunction transferFunction;
};

#endif
