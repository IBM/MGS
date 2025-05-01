// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LFPDataAnalyzer_H
#define LFPDataAnalyzer_H

#include "Mgs.h"
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
  virtual void setContributions(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LFPDataAnalyzerInAttrPSet* CG_inAttrPset, CG_LFPDataAnalyzerOutAttrPSet* CG_outAttrPset);
  LFPDataAnalyzer();
  virtual ~LFPDataAnalyzer();
  virtual void duplicate(std::unique_ptr<LFPDataAnalyzer>&& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
  virtual void duplicate(std::unique_ptr<CG_LFPDataAnalyzer>&& dup) const;
 private:
  std::ofstream* LFP_file;
  double normal_pdf(double x, double mean, double sigma);
};

#endif
