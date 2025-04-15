// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LeakyIAFUnit_H
#define LeakyIAFUnit_H

#include "Lens.h"
#include "CG_LeakyIAFUnit.h"
#include "rndm.h"

class LeakyIAFUnit : public CG_LeakyIAFUnit
{
 public:
  void initialize(RNG& rng);
  void update(RNG& rng);
  void threshold(RNG& rng);  
  virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LeakyIAFUnitInAttrPSet* CG_inAttrPset, CG_LeakyIAFUnitOutAttrPSet* CG_outAttrPset);
  virtual ~LeakyIAFUnit();
};

#endif
