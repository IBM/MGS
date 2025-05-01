// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NazeSORNUnitDataCollector_H
#define NazeSORNUnitDataCollector_H

#include "Mgs.h"
#include "CG_NazeSORNUnitDataCollector.h"
#include <memory>

class NazeSORNUnitDataCollector : public CG_NazeSORNUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_NazeSORNUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      NazeSORNUnitDataCollector();
      virtual ~NazeSORNUnitDataCollector();
      virtual void duplicate(std::unique_ptr<NazeSORNUnitDataCollector>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_NazeSORNUnitDataCollector>&& dup) const;
 private:
      std::ofstream* spikesFile;
};

#endif
