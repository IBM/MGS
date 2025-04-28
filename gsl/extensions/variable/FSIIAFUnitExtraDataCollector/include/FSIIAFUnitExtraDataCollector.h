// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
  virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitExtraDataCollectorInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitExtraDataCollectorOutAttrPSet* CG_outAttrPset);
  FSIIAFUnitExtraDataCollector();
  virtual ~FSIIAFUnitExtraDataCollector();
  virtual void duplicate(std::unique_ptr<FSIIAFUnitExtraDataCollector>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_FSIIAFUnitExtraDataCollector>&& dup) const;
 private:
  std::ofstream* voltages_file;
  std::ofstream* thresholds_file;
  std::ofstream* totalDriver_file;
  std::ofstream* totalIPSC_file;
  std::ofstream* totalGJ_file;
};

#endif
