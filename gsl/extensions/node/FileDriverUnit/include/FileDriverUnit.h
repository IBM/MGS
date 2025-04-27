// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FileDriverUnit_H
#define FileDriverUnit_H

#include "Lens.h"
#include "CG_FileDriverUnit.h"
#include "rndm.h"

class FileDriverUnit : public CG_FileDriverUnit
{
 public:
  void initialize(RNG& rng);
  void updateOutput(RNG& rng);
  virtual void setIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FileDriverUnitInAttrPSet* CG_inAttrPset, CG_FileDriverUnitOutAttrPSet* CG_outAttrPset);
  virtual ~FileDriverUnit();
};

#endif
