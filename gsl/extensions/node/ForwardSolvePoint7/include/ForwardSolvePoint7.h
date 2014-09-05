// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef ForwardSolvePoint7_H
#define ForwardSolvePoint7_H

#include "Lens.h"
#include "CG_ForwardSolvePoint7.h"
#include "rndm.h"

class ForwardSolvePoint7 : public CG_ForwardSolvePoint7
{
   public:
      void produceInitialState(RNG& rng);
      void produceInitialCoefficients(RNG& rng);
      void produceForwardSolution(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ForwardSolvePoint7InAttrPSet* CG_inAttrPset, CG_ForwardSolvePoint7OutAttrPSet* CG_outAttrPset);
      virtual ~ForwardSolvePoint7();
};

#endif
