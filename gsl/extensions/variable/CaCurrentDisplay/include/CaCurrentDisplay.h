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
      virtual void setUpPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaCurrentDisplayInAttrPSet* CG_inAttrPset, CG_CaCurrentDisplayOutAttrPSet* CG_outAttrPset);
      CaCurrentDisplay();
      virtual ~CaCurrentDisplay();
      virtual void duplicate(std::auto_ptr<CaCurrentDisplay>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_CaCurrentDisplay>& dup) const;
   private:
      std::ofstream* outFile = 0;
};

#endif
