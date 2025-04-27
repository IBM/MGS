// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
  virtual void setNeurotransmitterIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitOutAttrPSet* CG_outAttrPset);
  virtual void setPostSpikeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitOutAttrPSet* CG_outAttrPset);
  virtual ~SpineIAFUnit();
 private:
  double eCBsigmoid(double Ca);
  double eCBproduction(double Ca);
  double mGluR5modulation(double mGluR5);
};

#endif
