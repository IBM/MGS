// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CaConcentrationJunctionPoint_H
#define CaConcentrationJunctionPoint_H

#include "Lens.h"
#include "CG_CaConcentrationJunctionPoint.h"
#include "rndm.h"

class CaConcentrationJunctionPoint : public CG_CaConcentrationJunctionPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceCaConcentration(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionPointInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionPointOutAttrPSet* CG_outAttrPset);
      virtual ~CaConcentrationJunctionPoint();
};

#endif
