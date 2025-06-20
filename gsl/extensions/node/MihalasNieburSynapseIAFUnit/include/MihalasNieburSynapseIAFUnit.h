// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MihalasNieburSynapseIAFUnit_H
#define MihalasNieburSynapseIAFUnit_H

#include "Mgs.h"
#include "CG_MihalasNieburSynapseIAFUnit.h"
#include "rndm.h"

class MihalasNieburSynapseIAFUnit : public CG_MihalasNieburSynapseIAFUnit
{
 public:
  void initialize(RNG& rng);
  void update(RNG& rng);
  void threshold(RNG& rng);
  virtual void setAMPAIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MihalasNieburSynapseIAFUnitInAttrPSet* CG_inAttrPset, CG_MihalasNieburSynapseIAFUnitOutAttrPSet* CG_outAttrPset);
  virtual ~MihalasNieburSynapseIAFUnit();
};

#endif
