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

#ifndef ForwardSolvePoint1_H
#define ForwardSolvePoint1_H

#include "Lens.h"
#include "CG_ForwardSolvePoint1.h"
#include "rndm.h"

class ForwardSolvePoint1 : public CG_ForwardSolvePoint1
{
   public:
      void produceInitialState(RNG& rng);
      void produceInitialCoefficients(RNG& rng);
      void produceForwardSolution(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ForwardSolvePoint1InAttrPSet* CG_inAttrPset, CG_ForwardSolvePoint1OutAttrPSet* CG_outAttrPset);
      virtual ~ForwardSolvePoint1();
};

#endif
