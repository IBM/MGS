// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VoltageEndPoint_H
#define VoltageEndPoint_H

#include "Lens.h"
#include "CG_VoltageEndPoint.h"
#include "rndm.h"

class VoltageEndPoint : public CG_VoltageEndPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceSolvedVoltage(RNG& rng);
      void produceFinishedVoltage(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageEndPointInAttrPSet* CG_inAttrPset, CG_VoltageEndPointOutAttrPSet* CG_outAttrPset);
      virtual ~VoltageEndPoint();
};

#endif
