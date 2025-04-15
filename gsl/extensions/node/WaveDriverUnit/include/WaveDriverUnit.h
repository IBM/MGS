// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef WaveDriverUnit_H
#define WaveDriverUnit_H

#include "Lens.h"
#include "CG_WaveDriverUnit.h"
#include "rndm.h"

class WaveDriverUnit : public CG_WaveDriverUnit
{
 public:
  void initialize(RNG& rng);
  void update(RNG& rng);
  virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_WaveDriverUnitInAttrPSet* CG_inAttrPset, CG_WaveDriverUnitOutAttrPSet* CG_outAttrPset);
  virtual ~WaveDriverUnit();
};

#endif
