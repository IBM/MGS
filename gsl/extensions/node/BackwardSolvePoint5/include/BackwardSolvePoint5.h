// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BackwardSolvePoint5_H
#define BackwardSolvePoint5_H

#include "Mgs.h"
#include "CG_BackwardSolvePoint5.h"
#include "rndm.h"

class BackwardSolvePoint5 : public CG_BackwardSolvePoint5
{
   public:
      void produceInitialState(RNG& rng);
      void produceArea(RNG& rng);
      void produceBackwardSolution(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint5InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint5OutAttrPSet* CG_outAttrPset);
      virtual ~BackwardSolvePoint5();
};

#endif
