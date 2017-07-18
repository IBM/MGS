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

#ifndef CurrentDisplay_H
#define CurrentDisplay_H

#include "Lens.h"
#include "CG_CurrentDisplay.h"
#include <memory>
#include <fstream>

class CurrentDisplay : public CG_CurrentDisplay
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      virtual void setUpPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CurrentDisplayInAttrPSet* CG_inAttrPset, CG_CurrentDisplayOutAttrPSet* CG_outAttrPset);
      CurrentDisplay();
      virtual ~CurrentDisplay();
      virtual void duplicate(std::auto_ptr<CurrentDisplay>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_CurrentDisplay>& dup) const;
   private:
      std::ofstream* outFile;
};

#endif
