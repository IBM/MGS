// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NazeSORNUnitDataCollectorSuppl_H
#define NazeSORNUnitDataCollectorSuppl_H

#include "Lens.h"
#include "CG_NazeSORNUnitDataCollectorSuppl.h"
#include <memory>

class NazeSORNUnitDataCollectorSuppl : public CG_NazeSORNUnitDataCollectorSuppl
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNUnitDataCollectorSupplInAttrPSet* CG_inAttrPset, CG_NazeSORNUnitDataCollectorSupplOutAttrPSet* CG_outAttrPset);
      NazeSORNUnitDataCollectorSuppl();
      virtual ~NazeSORNUnitDataCollectorSuppl();
      virtual void duplicate(std::unique_ptr<NazeSORNUnitDataCollectorSuppl>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_NazeSORNUnitDataCollectorSuppl>& dup) const;
 private:
      std::ofstream* thresholdsFile;
};

#endif
