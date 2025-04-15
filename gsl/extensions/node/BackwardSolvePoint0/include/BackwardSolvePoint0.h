// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BackwardSolvePoint0_H
#define BackwardSolvePoint0_H

#include "Lens.h"
#include "CG_BackwardSolvePoint0.h"
#include "rndm.h"

class BackwardSolvePoint0 : public CG_BackwardSolvePoint0
{
   public:
      void produceInitialState(RNG& rng);
      void produceArea(RNG& rng);
      void produceBackwardSolution(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint0InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint0OutAttrPSet* CG_outAttrPset);
      virtual ~BackwardSolvePoint0();
};

#endif
