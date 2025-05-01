// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef VoltageAdapter_H
#define VoltageAdapter_H

#include "Mgs.h"
#include "CG_VoltageAdapter.h"
#include "rndm.h"

class VoltageAdapter : public CG_VoltageAdapter
{
   public:
      void produceInitialVoltage(RNG& rng);
      void produceVoltage(RNG& rng);
      void computeState(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageAdapterInAttrPSet* CG_inAttrPset, CG_VoltageAdapterOutAttrPSet* CG_outAttrPset);
      virtual ~VoltageAdapter();
};

#endif
