// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RabinovichWinnerlessUnitDataCollector_H
#define RabinovichWinnerlessUnitDataCollector_H

#include "Lens.h"
#include "CG_RabinovichWinnerlessUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class RabinovichWinnerlessUnitDataCollector : public CG_RabinovichWinnerlessUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_RabinovichWinnerlessUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_RabinovichWinnerlessUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      RabinovichWinnerlessUnitDataCollector();
      virtual ~RabinovichWinnerlessUnitDataCollector();
      virtual void duplicate(std::unique_ptr<RabinovichWinnerlessUnitDataCollector>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_RabinovichWinnerlessUnitDataCollector>& dup) const;
 private:
      std::ofstream *X_file;
};

#endif
