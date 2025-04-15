// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MegaSynapticCleft_H
#define MegaSynapticCleft_H

#include "Lens.h"
#include "CG_MegaSynapticCleft.h"
#include "rndm.h"

class MegaSynapticCleft : public CG_MegaSynapticCleft
{
   public:
      void produceInitialVoltage(RNG& rng);
      void produceVoltage(RNG& rng);
      void computeState(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MegaSynapticCleftInAttrPSet* CG_inAttrPset, CG_MegaSynapticCleftOutAttrPSet* CG_outAttrPset);
      virtual ~MegaSynapticCleft();
};

#endif
