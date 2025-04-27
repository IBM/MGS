// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BackwardSolvePoint3_H
#define BackwardSolvePoint3_H

#include "Lens.h"
#include "CG_BackwardSolvePoint3.h"
#include "rndm.h"

class BackwardSolvePoint3 : public CG_BackwardSolvePoint3
{
   public:
      void produceInitialState(RNG& rng);
      void produceArea(RNG& rng);
      void produceBackwardSolution(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint3InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint3OutAttrPSet* CG_outAttrPset);
      virtual ~BackwardSolvePoint3();
};

#endif
