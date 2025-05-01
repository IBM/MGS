// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MihalasNieburIAFUnitDataCollector_H
#define MihalasNieburIAFUnitDataCollector_H

#include "Mgs.h"
#include "CG_MihalasNieburIAFUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class MihalasNieburIAFUnitDataCollector : public CG_MihalasNieburIAFUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MihalasNieburIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_MihalasNieburIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      MihalasNieburIAFUnitDataCollector();
      virtual ~MihalasNieburIAFUnitDataCollector();
      virtual void duplicate(std::unique_ptr<MihalasNieburIAFUnitDataCollector>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_MihalasNieburIAFUnitDataCollector>&& dup) const;
 private:
      std::ofstream* voltage_file;
      std::ofstream* threshold_file;
      std::ofstream* spike_file;
};

#endif
