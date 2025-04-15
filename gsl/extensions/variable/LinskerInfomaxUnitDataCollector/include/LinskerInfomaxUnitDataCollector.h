// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LinskerInfomaxUnitDataCollector_H
#define LinskerInfomaxUnitDataCollector_H

#include "Lens.h"
#include "CG_LinskerInfomaxUnitDataCollector.h"
#include <memory>

class LinskerInfomaxUnitDataCollector : public CG_LinskerInfomaxUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LinskerInfomaxUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_LinskerInfomaxUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      LinskerInfomaxUnitDataCollector();
      virtual ~LinskerInfomaxUnitDataCollector();
      virtual void duplicate(std::unique_ptr<LinskerInfomaxUnitDataCollector>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_LinskerInfomaxUnitDataCollector>& dup) const;
 private:
      std::ofstream* file;
};

#endif
