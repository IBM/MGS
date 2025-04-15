// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BoutonIAFUnitDataCollector_H
#define BoutonIAFUnitDataCollector_H

#include "Lens.h"
#include "CG_BoutonIAFUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class BoutonIAFUnitDataCollector : public CG_BoutonIAFUnitDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BoutonIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_BoutonIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  BoutonIAFUnitDataCollector();
  virtual ~BoutonIAFUnitDataCollector();
  virtual void duplicate(std::unique_ptr<BoutonIAFUnitDataCollector>& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>& dup) const;
  virtual void duplicate(std::unique_ptr<CG_BoutonIAFUnitDataCollector>& dup) const;
 private:
  std::ofstream* neurotransmitter_file;
  std::ofstream* availableNeurotransmitter_file;
  std::ofstream* CB1R_file;
  std::ofstream* CB1Runbound_file;
  std::ofstream* CB1Rcurrent_file;
};

#endif
