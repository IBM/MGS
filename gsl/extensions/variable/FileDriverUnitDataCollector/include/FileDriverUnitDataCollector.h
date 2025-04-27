// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FileDriverUnitDataCollector_H
#define FileDriverUnitDataCollector_H

#include "Lens.h"
#include "CG_FileDriverUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class FileDriverUnitDataCollector : public CG_FileDriverUnitDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FileDriverUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_FileDriverUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  FileDriverUnitDataCollector();
  virtual ~FileDriverUnitDataCollector();
  virtual void duplicate(std::unique_ptr<FileDriverUnitDataCollector>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_FileDriverUnitDataCollector>&& dup) const;
 private:
  std::ofstream* output_file;
};

#endif
