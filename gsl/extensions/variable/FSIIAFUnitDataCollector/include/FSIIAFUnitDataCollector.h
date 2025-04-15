// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FSIIAFUnitDataCollector_H
#define FSIIAFUnitDataCollector_H

#include "Lens.h"
#include "CG_FSIIAFUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class FSIIAFUnitDataCollector : public CG_FSIIAFUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      FSIIAFUnitDataCollector();
      virtual ~FSIIAFUnitDataCollector();
      virtual void duplicate(std::unique_ptr<FSIIAFUnitDataCollector>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_FSIIAFUnitDataCollector>& dup) const;
 private:
  std::ofstream* spikes_file;
};

#endif
