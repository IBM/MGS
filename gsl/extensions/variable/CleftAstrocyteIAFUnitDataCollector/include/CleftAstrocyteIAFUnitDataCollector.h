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

#ifndef CleftAstrocyteIAFUnitDataCollector_H
#define CleftAstrocyteIAFUnitDataCollector_H

#include "Lens.h"
#include "CG_CleftAstrocyteIAFUnitDataCollector.h"
#include <memory>

class CleftAstrocyteIAFUnitDataCollector : public CG_CleftAstrocyteIAFUnitDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  CleftAstrocyteIAFUnitDataCollector();
  virtual ~CleftAstrocyteIAFUnitDataCollector();
  virtual void duplicate(std::auto_ptr<CleftAstrocyteIAFUnitDataCollector>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_CleftAstrocyteIAFUnitDataCollector>& dup) const;
 private:
  std::ofstream* neurotransmitter_file;
  std::ofstream* eCB_file;
};

#endif
