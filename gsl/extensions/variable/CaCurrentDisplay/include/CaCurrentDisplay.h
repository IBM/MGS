// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CaCurrentDisplay_H
#define CaCurrentDisplay_H

#include "Lens.h"
#include "CG_CaCurrentDisplay.h"
#include <memory>
#include <fstream>

class CaCurrentDisplay : public CG_CaCurrentDisplay
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void setUpPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaCurrentDisplayInAttrPSet* CG_inAttrPset, CG_CaCurrentDisplayOutAttrPSet* CG_outAttrPset);
      CaCurrentDisplay();
      virtual ~CaCurrentDisplay();
      virtual void duplicate(std::unique_ptr<CaCurrentDisplay>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_CaCurrentDisplay>&& dup) const;
   private:
      std::ofstream* outFile = 0;
};

#endif
