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

#ifndef TraubIAFUnitExtraDataCollector_H
#define TraubIAFUnitExtraDataCollector_H

#include "Lens.h"
#include "CG_TraubIAFUnitExtraDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class TraubIAFUnitExtraDataCollector : public CG_TraubIAFUnitExtraDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_TraubIAFUnitExtraDataCollectorInAttrPSet* CG_inAttrPset, CG_TraubIAFUnitExtraDataCollectorOutAttrPSet* CG_outAttrPset);
  TraubIAFUnitExtraDataCollector();
  virtual ~TraubIAFUnitExtraDataCollector();
  virtual void duplicate(std::auto_ptr<TraubIAFUnitExtraDataCollector>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_TraubIAFUnitExtraDataCollector>& dup) const;
 private:
  std::ofstream* voltages_file;
  std::ofstream* thresholds_file;
  std::ofstream* totalDriver_file;
  std::ofstream* totalIPSC_file;
  std::ofstream* totalGJ_file;
};

#endif
