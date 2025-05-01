// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MihalasNieburSynapseIAFUnitDataCollector_H
#define MihalasNieburSynapseIAFUnitDataCollector_H

#include "Mgs.h"
#include "CG_MihalasNieburSynapseIAFUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class MihalasNieburSynapseIAFUnitDataCollector : public CG_MihalasNieburSynapseIAFUnitDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollectionSpikes(Trigger* trigger, NDPairList* ndPairList);
  virtual void dataCollectionOther(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MihalasNieburSynapseIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_MihalasNieburSynapseIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  MihalasNieburSynapseIAFUnitDataCollector();
  virtual ~MihalasNieburSynapseIAFUnitDataCollector();
  virtual void duplicate(std::unique_ptr<MihalasNieburSynapseIAFUnitDataCollector>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_MihalasNieburSynapseIAFUnitDataCollector>&& dup) const;
 private:
  std::ofstream* voltage_file;
  std::ofstream* threshold_file;
  std::ofstream* spike_file;
  std::ofstream* spikevoltage_file;
};

#endif
