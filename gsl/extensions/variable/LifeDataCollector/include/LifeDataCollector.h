// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LifeDataCollector_H
#define LifeDataCollector_H

#include "Lens.h"
#include "CG_LifeDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

class LifeDataCollector : public CG_LifeDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LifeDataCollectorInAttrPSet* CG_inAttrPset, CG_LifeDataCollectorOutAttrPSet* CG_outAttrPset);
      LifeDataCollector();
      virtual ~LifeDataCollector();
      virtual void duplicate(std::unique_ptr<LifeDataCollector>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_LifeDataCollector>&& dup) const;

 private:
      std::ofstream* file;
};

#endif
