// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SpineIAFUnitDataCollector_H
#define SpineIAFUnitDataCollector_H

#include "Lens.h"
#include "CG_SpineIAFUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class SpineIAFUnitDataCollector : public CG_SpineIAFUnitDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  SpineIAFUnitDataCollector();
  virtual ~SpineIAFUnitDataCollector();
  virtual void duplicate(std::unique_ptr<SpineIAFUnitDataCollector>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_SpineIAFUnitDataCollector>&& dup) const;
 private:
  std::ofstream* AMPA_file;
  std::ofstream* mGluR5_file;
  std::ofstream* mGluR5modulation_file;
  std::ofstream* NMDARopen_file;
  std::ofstream* NMDARCacurrent_file;
  std::ofstream* Ca_file;
  std::ofstream* eCBproduction_file; // just for the production function
  std::ofstream* eCB_file;
  double eCBsigmoid(double Ca);
  double eCBproduction(double Ca);
  double mGluR5modulation(double mGluR5);
};

#endif
