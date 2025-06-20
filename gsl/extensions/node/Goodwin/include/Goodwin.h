// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Goodwin_H
#define Goodwin_H

#include "Mgs.h"
#include "CG_Goodwin.h"
#include "rndm.h"

class Goodwin : public CG_Goodwin
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void setInIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GoodwinInAttrPSet* CG_inAttrPset, CG_GoodwinOutAttrPSet* CG_outAttrPset);
      virtual ~Goodwin();
   // Model specific additional functions
   private:
      double Cannabinoids_Y_minus_eCB_sigmoid(double Y_minus_eCB);
};

#endif
