// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef BackwardSolvePoint1_H
#define BackwardSolvePoint1_H

#include "Lens.h"
#include "CG_BackwardSolvePoint1.h"
#include "rndm.h"

class BackwardSolvePoint1 : public CG_BackwardSolvePoint1
{
   public:
      void produceInitialState(RNG& rng);
      void produceArea(RNG& rng);
      void produceBackwardSolution(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint1InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint1OutAttrPSet* CG_outAttrPset);
      virtual ~BackwardSolvePoint1();
};

#endif
