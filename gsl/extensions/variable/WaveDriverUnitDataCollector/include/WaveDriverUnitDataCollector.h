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

#ifndef WaveDriverUnitDataCollector_H
#define WaveDriverUnitDataCollector_H

#include "Lens.h"
#include "CG_WaveDriverUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class WaveDriverUnitDataCollector : public CG_WaveDriverUnitDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_WaveDriverUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_WaveDriverUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  WaveDriverUnitDataCollector();
  virtual ~WaveDriverUnitDataCollector();
  virtual void duplicate(std::unique_ptr<WaveDriverUnitDataCollector>& dup) const;
  virtual void duplicate(std::unique_ptr<Variable>& dup) const;
  virtual void duplicate(std::unique_ptr<CG_WaveDriverUnitDataCollector>& dup) const;
 private:
  std::ofstream* wave_file;
};

#endif
