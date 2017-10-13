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
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GoodwinDataCollectorInAttrPSet* CG_inAttrPset, CG_GoodwinDataCollectorOutAttrPSet* CG_outAttrPset);
  GoodwinDataCollector();
  virtual ~GoodwinDataCollector();
  virtual void duplicate(std::auto_ptr<GoodwinDataCollector>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_GoodwinDataCollector>& dup) const;
 private:
  std::ofstream* X_file;  
  std::ofstream* Y_file;  
  std::ofstream* Z_file;  
};

#endif
