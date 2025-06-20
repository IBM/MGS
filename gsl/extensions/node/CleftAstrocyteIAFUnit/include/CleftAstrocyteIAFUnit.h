// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CleftAstrocyteIAFUnit_H
#define CleftAstrocyteIAFUnit_H

#include "Mgs.h"
#include "CG_CleftAstrocyteIAFUnit.h"
#include "rndm.h"

class CleftAstrocyteIAFUnit : public CG_CleftAstrocyteIAFUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void setNeurotransmitterIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual void seteCBIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitOutAttrPSet* CG_outAttrPset);
      virtual ~CleftAstrocyteIAFUnit();
};

#endif
