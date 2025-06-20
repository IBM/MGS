// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef IzhikUnitDataCollector_H
#define IzhikUnitDataCollector_H

#include "Mgs.h"
#include "CG_IzhikUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class IzhikUnitDataCollector : public CG_IzhikUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void dataCollectionSpike(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IzhikUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_IzhikUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      IzhikUnitDataCollector();
      virtual ~IzhikUnitDataCollector();
      virtual void duplicate(std::unique_ptr<IzhikUnitDataCollector>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_IzhikUnitDataCollector>&& dup) const;


 private:
      std::ofstream* voltage_file;
      std::ofstream* spike_file;
      std::ofstream* trans_file;
};

#endif
