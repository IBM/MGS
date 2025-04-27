// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ZhengSORNUnitDataCollector_H
#define ZhengSORNUnitDataCollector_H

#include "Lens.h"
#include "CG_ZhengSORNUnitDataCollector.h"
#include <memory>

class ZhengSORNUnitDataCollector : public CG_ZhengSORNUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_ZhengSORNUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      ZhengSORNUnitDataCollector();
      virtual ~ZhengSORNUnitDataCollector();
      virtual void duplicate(std::unique_ptr<ZhengSORNUnitDataCollector>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ZhengSORNUnitDataCollector>&& dup) const;
 private:
      std::ofstream* spikesFile;
};

#endif
