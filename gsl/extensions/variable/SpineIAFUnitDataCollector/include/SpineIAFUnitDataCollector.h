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

#ifndef SpineIAFUnitDataCollector_H
#define SpineIAFUnitDataCollector_H

#include "Lens.h"
#include "CG_SpineIAFUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class SpineIAFUnitDataCollector : public CG_SpineIAFUnitDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  SpineIAFUnitDataCollector();
  virtual ~SpineIAFUnitDataCollector();
  virtual void duplicate(std::auto_ptr<SpineIAFUnitDataCollector>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_SpineIAFUnitDataCollector>& dup) const;
 private:
  std::ofstream* AMPA_file;
  std::ofstream* mGluR5_file;
  std::ofstream* mGluR5modulation_file;
  std::ofstream* NMDARopen_file;
  std::ofstream* NMDARCacurrent_file;
  std::ofstream* Ca_file;
  std::ofstream* eCBproduction_file; // just for the production function
  std::ofstream* eCB_file;
  double eCBsigmoid(double Ca);
  double eCBproduction(double Ca);
  double mGluR5modulation(double mGluR5);
};

#endif
