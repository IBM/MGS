// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef ToneUnit_H
#define ToneUnit_H

#include "Mgs.h"
#include "CG_ToneUnit.h"
#include "rndm.h"

class ToneUnit : public CG_ToneUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void setIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ToneUnitInAttrPSet* CG_inAttrPset, CG_ToneUnitOutAttrPSet* CG_outAttrPset);
      virtual ~ToneUnit();
};

#endif
