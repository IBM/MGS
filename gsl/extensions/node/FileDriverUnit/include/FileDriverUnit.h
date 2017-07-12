// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
  virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FileDriverUnitInAttrPSet* CG_inAttrPset, CG_FileDriverUnitOutAttrPSet* CG_outAttrPset);
  virtual ~FileDriverUnit();
};

#endif
