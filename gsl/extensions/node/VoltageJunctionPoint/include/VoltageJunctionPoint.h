// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VoltageJunctionPoint_H
#define VoltageJunctionPoint_H

#include "Lens.h"
#include "CG_VoltageJunctionPoint.h"
#include "rndm.h"

class VoltageJunctionPoint : public CG_VoltageJunctionPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceVoltage(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageJunctionPointInAttrPSet* CG_inAttrPset, CG_VoltageJunctionPointOutAttrPSet* CG_outAttrPset);
      virtual ~VoltageJunctionPoint();
};

#endif
