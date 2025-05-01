// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NodeSetSPMVariable_H
#define NodeSetSPMVariable_H
#include "Mgs.h"

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
      virtual void duplicate(std::unique_ptr<NodeSetSPMVariable>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_NodeSetSPMVariable>&& dup) const;

   private:
      unsigned dimx;
      unsigned dimy;
};

#endif
