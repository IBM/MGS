// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PoissonIAFUnitDataCollector_H
#define PoissonIAFUnitDataCollector_H

#include "Lens.h"
#include "CG_PoissonIAFUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class PoissonIAFUnitDataCollector : public CG_PoissonIAFUnitDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_PoissonIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_PoissonIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  PoissonIAFUnitDataCollector();
  virtual ~PoissonIAFUnitDataCollector();
  virtual void duplicate(std::unique_ptr<PoissonIAFUnitDataCollector>& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>& dup) const;
  virtual void duplicate(std::unique_ptr<CG_PoissonIAFUnitDataCollector>& dup) const;
 private:
  std::ofstream* spikes_file;
};

#endif
