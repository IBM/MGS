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

#ifndef MihalasNieburSynapseIAFUnitDataCollector_H
#define MihalasNieburSynapseIAFUnitDataCollector_H

#include "Lens.h"
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
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MihalasNieburSynapseIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_MihalasNieburSynapseIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  MihalasNieburSynapseIAFUnitDataCollector();
  virtual ~MihalasNieburSynapseIAFUnitDataCollector();
  virtual void duplicate(std::auto_ptr<MihalasNieburSynapseIAFUnitDataCollector>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_MihalasNieburSynapseIAFUnitDataCollector>& dup) const;
 private:
  std::ofstream* voltage_file;
  std::ofstream* threshold_file;
  std::ofstream* spike_file;
  std::ofstream* spikevoltage_file;
};

#endif
