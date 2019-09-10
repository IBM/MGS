// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2018
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef FSIIAFUnitExtraDataCollector_H
#define FSIIAFUnitExtraDataCollector_H

#include "Lens.h"
#include "CG_FSIIAFUnitExtraDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class FSIIAFUnitExtraDataCollector : public CG_FSIIAFUnitExtraDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitExtraDataCollectorInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitExtraDataCollectorOutAttrPSet* CG_outAttrPset);
  FSIIAFUnitExtraDataCollector();
  virtual ~FSIIAFUnitExtraDataCollector();
  virtual void duplicate(std::unique_ptr<FSIIAFUnitExtraDataCollector>& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>& dup) const;
  virtual void duplicate(std::unique_ptr<CG_FSIIAFUnitExtraDataCollector>& dup) const;
 private:
  std::ofstream* voltages_file;
  std::ofstream* thresholds_file;
  std::ofstream* totalDriver_file;
  std::ofstream* totalIPSC_file;
  std::ofstream* totalGJ_file;
};

#endif
