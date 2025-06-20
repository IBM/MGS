// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef AtasoyNFUnit_H
#define AtasoyNFUnit_H

#include "Mgs.h"
#include "CG_AtasoyNFUnit.h"
#include "rndm.h"

class AtasoyNFUnit : public CG_AtasoyNFUnit
{
   public:
      void initialize(RNG& rng);
      void diffusion(RNG& rng);
      void reaction(RNG& rng);
      virtual bool checkForConnection(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_AtasoyNFUnitInAttrPSet* CG_inAttrPset, CG_AtasoyNFUnitOutAttrPSet* CG_outAttrPset);
      virtual ~AtasoyNFUnit();
};

#endif
