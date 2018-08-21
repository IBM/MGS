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
  virtual void duplicate(std::auto_ptr<BoutonIAFUnitDataCollector>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_BoutonIAFUnitDataCollector>& dup) const;
 private:
  std::ofstream* neurotransmitter_file;
  std::ofstream* availableNeurotransmitter_file;
  std::ofstream* CB1R_file;
  std::ofstream* CB1Runbound_file;
  std::ofstream* CB1Rcurrent_file;
};

#endif
