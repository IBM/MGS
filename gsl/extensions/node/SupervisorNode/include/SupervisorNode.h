#ifndef SupervisorNode_H
#define SupervisorNode_H

#include "Lens.h"
#include "CG_SupervisorNode.h"
#include "rndm.h"
<<<<<<< HEAD
=======
#include "TransferFunction.h"
>>>>>>> Adding DNN model suite.

class SupervisorNode : public CG_SupervisorNode
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual ~SupervisorNode();
<<<<<<< HEAD
=======

   private:
      TransferFunction transferFunction;
      
>>>>>>> Adding DNN model suite.
};

#endif
