// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ConductanceDisplay_H
#define ConductanceDisplay_H

#include "Lens.h"
#include "CG_ConductanceDisplay.h"
#include <memory>
#include <fstream>

class ConductanceDisplay : public CG_ConductanceDisplay
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void setUpPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ConductanceDisplayInAttrPSet* CG_inAttrPset, CG_ConductanceDisplayOutAttrPSet* CG_outAttrPset);
      ConductanceDisplay();
      virtual ~ConductanceDisplay();
      virtual void duplicate(std::unique_ptr<ConductanceDisplay>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ConductanceDisplay>&& dup) const;
   private:
      std::ofstream* outFile = 0;
};

#endif
