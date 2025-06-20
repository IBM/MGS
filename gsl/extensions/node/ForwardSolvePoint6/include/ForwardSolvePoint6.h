// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ForwardSolvePoint6_H
#define ForwardSolvePoint6_H

#include "Mgs.h"
#include "CG_ForwardSolvePoint6.h"
#include "rndm.h"

class ForwardSolvePoint6 : public CG_ForwardSolvePoint6
{
   public:
      void produceInitialState(RNG& rng);
      void produceInitialCoefficients(RNG& rng);
      void produceForwardSolution(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ForwardSolvePoint6InAttrPSet* CG_inAttrPset, CG_ForwardSolvePoint6OutAttrPSet* CG_outAttrPset);
      virtual ~ForwardSolvePoint6();
};

#endif
