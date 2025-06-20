// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef GatedThalamoCorticalUnitDataCollector_H
#define GatedThalamoCorticalUnitDataCollector_H

#include "Mgs.h"
#include "CG_GatedThalamoCorticalUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class GatedThalamoCorticalUnitDataCollector : public CG_GatedThalamoCorticalUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GatedThalamoCorticalUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_GatedThalamoCorticalUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      GatedThalamoCorticalUnitDataCollector();
      virtual ~GatedThalamoCorticalUnitDataCollector();
      virtual void duplicate(std::unique_ptr<GatedThalamoCorticalUnitDataCollector>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_GatedThalamoCorticalUnitDataCollector>&& dup) const;

 private:
      std::ofstream* file;
      std::ofstream* yfile;
};

#endif
