// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CleftAstrocyteIAFUnitDataCollector_H
#define CleftAstrocyteIAFUnitDataCollector_H

#include "Mgs.h"
#include "CG_CleftAstrocyteIAFUnitDataCollector.h"
#include <memory>

class CleftAstrocyteIAFUnitDataCollector : public CG_CleftAstrocyteIAFUnitDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  CleftAstrocyteIAFUnitDataCollector();
  virtual ~CleftAstrocyteIAFUnitDataCollector();
  virtual void duplicate(std::unique_ptr<CleftAstrocyteIAFUnitDataCollector>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_CleftAstrocyteIAFUnitDataCollector>&& dup) const;
 private:
  std::ofstream* neurotransmitter_file;
  std::ofstream* eCB_file;
};

#endif
