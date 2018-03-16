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

#ifndef ZhengSORNUnitDataCollectorSuppl_H
#define ZhengSORNUnitDataCollectorSuppl_H

#include "Lens.h"
#include "CG_ZhengSORNUnitDataCollectorSuppl.h"
#include <memory>

class ZhengSORNUnitDataCollectorSuppl : public CG_ZhengSORNUnitDataCollectorSuppl
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNUnitDataCollectorSupplInAttrPSet* CG_inAttrPset, CG_ZhengSORNUnitDataCollectorSupplOutAttrPSet* CG_outAttrPset);
      ZhengSORNUnitDataCollectorSuppl();
      virtual ~ZhengSORNUnitDataCollectorSuppl();
      virtual void duplicate(std::auto_ptr<ZhengSORNUnitDataCollectorSuppl>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ZhengSORNUnitDataCollectorSuppl>& dup) const;
 private:
      std::ofstream* thresholdsFile;
};

#endif
