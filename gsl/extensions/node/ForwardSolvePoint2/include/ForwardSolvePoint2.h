// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ForwardSolvePoint2_H
#define ForwardSolvePoint2_H

#include "Lens.h"
#include "CG_ForwardSolvePoint2.h"
#include "rndm.h"

class ForwardSolvePoint2 : public CG_ForwardSolvePoint2
{
   public:
      void produceInitialState(RNG& rng);
      void produceInitialCoefficients(RNG& rng);
      void produceForwardSolution(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ForwardSolvePoint2InAttrPSet* CG_inAttrPset, CG_ForwardSolvePoint2OutAttrPSet* CG_outAttrPset);
      virtual ~ForwardSolvePoint2();
};

#endif
