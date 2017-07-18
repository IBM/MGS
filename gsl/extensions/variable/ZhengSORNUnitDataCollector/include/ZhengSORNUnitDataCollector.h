// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_ZhengSORNUnitDataCollectorOutAttrPSet* CG_outAttrPset);
      ZhengSORNUnitDataCollector();
      virtual ~ZhengSORNUnitDataCollector();
      virtual void duplicate(std::auto_ptr<ZhengSORNUnitDataCollector>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ZhengSORNUnitDataCollector>& dup) const;
 private:
      std::ofstream* file;
};

#endif
