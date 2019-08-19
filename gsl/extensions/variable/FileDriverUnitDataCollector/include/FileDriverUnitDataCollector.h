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
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FileDriverUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_FileDriverUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  FileDriverUnitDataCollector();
  virtual ~FileDriverUnitDataCollector();
  virtual void duplicate(std::unique_ptr<FileDriverUnitDataCollector>& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>& dup) const;
  virtual void duplicate(std::unique_ptr<CG_FileDriverUnitDataCollector>& dup) const;
 private:
  std::ofstream* output_file;
};

#endif
