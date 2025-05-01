// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PreSynapticPoint_H
#define PreSynapticPoint_H

#include "Mgs.h"
#include "CG_PreSynapticPoint.h"
#include "rndm.h"

class PreSynapticPoint : public CG_PreSynapticPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceState(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_PreSynapticPointInAttrPSet* CG_inAttrPset, CG_PreSynapticPointOutAttrPSet* CG_outAttrPset);
      virtual ~PreSynapticPoint();
};

#endif
