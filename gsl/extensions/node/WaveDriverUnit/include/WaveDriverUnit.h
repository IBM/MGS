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
