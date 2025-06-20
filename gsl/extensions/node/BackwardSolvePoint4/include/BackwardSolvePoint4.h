// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BackwardSolvePoint4_H
#define BackwardSolvePoint4_H

#include "Mgs.h"
#include "CG_BackwardSolvePoint4.h"
#include "rndm.h"

class BackwardSolvePoint4 : public CG_BackwardSolvePoint4
{
   public:
      void produceInitialState(RNG& rng);
      void produceArea(RNG& rng);
      void produceBackwardSolution(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint4InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint4OutAttrPSet* CG_outAttrPset);
      virtual ~BackwardSolvePoint4();
};

#endif
