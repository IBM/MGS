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

#ifndef Goodwin_H
#define Goodwin_H

#include "Lens.h"
#include "CG_Goodwin.h"
#include "rndm.h"

class Goodwin : public CG_Goodwin
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void setInIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GoodwinInAttrPSet* CG_inAttrPset, CG_GoodwinOutAttrPSet* CG_outAttrPset);
      virtual ~Goodwin();
};

#endif
