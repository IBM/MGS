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

#ifndef LFPDataAnalyzer_H
#define LFPDataAnalyzer_H

#include "Lens.h"
#include "CG_LFPDataAnalyzer.h"
#include <memory>
#include <fstream>
#include <iostream>

class LFPDataAnalyzer : public CG_LFPDataAnalyzer
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LFPDataAnalyzerInAttrPSet* CG_inAttrPset, CG_LFPDataAnalyzerOutAttrPSet* CG_outAttrPset);
  LFPDataAnalyzer();
  virtual ~LFPDataAnalyzer();
  virtual void duplicate(std::unique_ptr<LFPDataAnalyzer>& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>& dup) const;
  virtual void duplicate(std::unique_ptr<CG_LFPDataAnalyzer>& dup) const;
 private:
  std::ofstream* LFP_file;
  double normal_pdf(double x, double mean, double sigma);
};

#endif
