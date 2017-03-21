// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef BackwardSolvePoint4_H
#define BackwardSolvePoint4_H

#include "Lens.h"
#include "CG_BackwardSolvePoint4.h"
#include "rndm.h"

class BackwardSolvePoint4 : public CG_BackwardSolvePoint4
{
   public:
      void produceInitialState(RNG& rng);
      void produceArea(RNG& rng);
      void produceBackwardSolution(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BackwardSolvePoint4InAttrPSet* CG_inAttrPset, CG_BackwardSolvePoint4OutAttrPSet* CG_outAttrPset);
      virtual ~BackwardSolvePoint4();
};

#endif
