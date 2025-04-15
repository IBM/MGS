// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ForwardSolvePoint5_H
#define ForwardSolvePoint5_H

#include "Lens.h"
#include "CG_ForwardSolvePoint5.h"
#include "rndm.h"

class ForwardSolvePoint5 : public CG_ForwardSolvePoint5
{
   public:
      void produceInitialState(RNG& rng);
      void produceInitialCoefficients(RNG& rng);
      void produceForwardSolution(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ForwardSolvePoint5InAttrPSet* CG_inAttrPset, CG_ForwardSolvePoint5OutAttrPSet* CG_outAttrPset);
      virtual ~ForwardSolvePoint5();
};

#endif
