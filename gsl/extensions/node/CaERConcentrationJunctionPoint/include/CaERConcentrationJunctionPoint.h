// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef CaERConcentrationJunctionPoint_H
#define CaERConcentrationJunctionPoint_H

#include "Mgs.h"
#include "CG_CaERConcentrationJunctionPoint.h"
#include "rndm.h"

class CaERConcentrationJunctionPoint : public CG_CaERConcentrationJunctionPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceCaConcentration(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationJunctionPointInAttrPSet* CG_inAttrPset, CG_CaERConcentrationJunctionPointOutAttrPSet* CG_outAttrPset);
      virtual ~CaERConcentrationJunctionPoint();
};

#endif
