// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef LypCollector_H
#define LypCollector_H

#include "Mgs.h"
#include "CG_LypCollector.h"
#include "rndm.h"

class LypCollector : public CG_LypCollector
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void setIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LypCollectorInAttrPSet* CG_inAttrPset, CG_LypCollectorOutAttrPSet* CG_outAttrPset);
      virtual ~LypCollector();
};

#endif
