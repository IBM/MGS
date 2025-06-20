// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ZhengSORNUnitDataCollectorSuppl_H
#define ZhengSORNUnitDataCollectorSuppl_H

#include "Mgs.h"
#include "CG_ZhengSORNUnitDataCollectorSuppl.h"
#include <memory>

class ZhengSORNUnitDataCollectorSuppl : public CG_ZhengSORNUnitDataCollectorSuppl
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNUnitDataCollectorSupplInAttrPSet* CG_inAttrPset, CG_ZhengSORNUnitDataCollectorSupplOutAttrPSet* CG_outAttrPset);
      ZhengSORNUnitDataCollectorSuppl();
      virtual ~ZhengSORNUnitDataCollectorSuppl();
      virtual void duplicate(std::unique_ptr<ZhengSORNUnitDataCollectorSuppl>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ZhengSORNUnitDataCollectorSuppl>&& dup) const;
 private:
      std::ofstream* thresholdsFile;
};

#endif
