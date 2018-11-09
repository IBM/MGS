// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef NodeSetSPMVariable_H
#define NodeSetSPMVariable_H
#include "Lens.h"

#include "CG_NodeSetSPMVariable.h"
#include <memory>
#include "ShallowArray.h"

class NodeSetSPMVariable : public CG_NodeSetSPMVariable
{
   public:
      //CUDA_CALLABLE 
      virtual void initialize(RNG&);
      virtual void dca(Trigger* trigger, NDPairList* ndPairList);
      NodeSetSPMVariable();
      virtual ~NodeSetSPMVariable();
      virtual void duplicate(std::unique_ptr<NodeSetSPMVariable>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_NodeSetSPMVariable>& dup) const;

   private:
      unsigned dimx;
      unsigned dimy;
};

#endif
