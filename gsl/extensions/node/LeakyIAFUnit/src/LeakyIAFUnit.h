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
