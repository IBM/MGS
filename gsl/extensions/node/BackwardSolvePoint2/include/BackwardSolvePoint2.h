// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BackwardSolvePoint2_H
#define BackwardSolvePoint2_H

#include "Lens.h"
#include "CG_BackwardSolvePoint2.h"
#include "rndm.h"

class BackwardSolvePoint2 : public CG_BackwardSolvePoint2
{
   public:
      void produceInitialState(RNG& rng);
      void produceArea(RNG& rng);
      void produceBackwardSolution(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint2InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint2OutAttrPSet* CG_outAttrPset);
      virtual ~BackwardSolvePoint2();
};

#endif
