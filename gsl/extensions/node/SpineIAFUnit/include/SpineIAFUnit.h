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

#ifndef SpineIAFUnit_H
#define SpineIAFUnit_H

#include "Lens.h"
#include "CG_SpineIAFUnit.h"
#include "rndm.h"

class SpineIAFUnit : public CG_SpineIAFUnit
{
 public:
  void initialize(RNG& rng);
  void update(RNG& rng);
  void outputWeights(std::ofstream& fs);
  virtual void setGlutamateIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitOutAttrPSet* CG_outAttrPset);
  virtual void setPostSpikeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitOutAttrPSet* CG_outAttrPset);
  virtual ~SpineIAFUnit();
 private:
  double ECBsigmoid(double Ca);
  double ECBproduction(double Ca);
<<<<<<< HEAD
  double mGluRmodulation(double mGluR);
=======
  double mGluR5modulation(double mGluR5);
>>>>>>> origin/team-A
};

#endif
