// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LeakyIAFUnitDataCollector_H
#define LeakyIAFUnitDataCollector_H

#include "Mgs.h"
#include "CG_LeakyIAFUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class LeakyIAFUnitDataCollector : public CG_LeakyIAFUnitDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollectionSpikes(Trigger* trigger, NDPairList* ndPairList);
  virtual void dataCollectionOther(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LeakyIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_LeakyIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  LeakyIAFUnitDataCollector();
  virtual ~LeakyIAFUnitDataCollector();
  virtual void duplicate(std::unique_ptr<LeakyIAFUnitDataCollector>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_LeakyIAFUnitDataCollector>&& dup) const;
 private:
  std::ofstream* voltage_file;
  std::ofstream* spike_file;
};

#endif
