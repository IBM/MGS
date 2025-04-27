// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GoodwinDataCollector_H
#define GoodwinDataCollector_H

#include "Lens.h"
#include "CG_GoodwinDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class GoodwinDataCollector : public CG_GoodwinDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GoodwinDataCollectorInAttrPSet* CG_inAttrPset, CG_GoodwinDataCollectorOutAttrPSet* CG_outAttrPset);
  GoodwinDataCollector();
  virtual ~GoodwinDataCollector();
  virtual void duplicate(std::unique_ptr<GoodwinDataCollector>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) dup) const;
  virtual void duplicate(std::unique_ptr<CG_GoodwinDataCollector>&& dup) const;
 private:
  std::ofstream* X_file;  
  std::ofstream* Y_file;  
  std::ofstream* Z_file;  
};

#endif
