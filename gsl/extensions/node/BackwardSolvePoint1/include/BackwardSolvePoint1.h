// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BackwardSolvePoint1_H
#define BackwardSolvePoint1_H

#include "Mgs.h"
#include "CG_BackwardSolvePoint1.h"
#include "rndm.h"

class BackwardSolvePoint1 : public CG_BackwardSolvePoint1
{
   public:
      void produceInitialState(RNG& rng);
      void produceArea(RNG& rng);
      void produceBackwardSolution(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint1InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint1OutAttrPSet* CG_outAttrPset);
      virtual ~BackwardSolvePoint1();
};

#endif
