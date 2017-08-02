// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef LeakyIAFUnitDataCollector_H
#define LeakyIAFUnitDataCollector_H

#include "Lens.h"
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
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LeakyIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_LeakyIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  LeakyIAFUnitDataCollector();
  virtual ~LeakyIAFUnitDataCollector();
  virtual void duplicate(std::auto_ptr<LeakyIAFUnitDataCollector>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_LeakyIAFUnitDataCollector>& dup) const;
 private:
  std::ofstream* voltage_file;
  std::ofstream* spike_file;
};

#endif
