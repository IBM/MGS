// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef BengioRatePyramidal_H
#define BengioRatePyramidal_H

#include "Mgs.h"
#include "CG_BengioRatePyramidal.h"
#include "rndm.h"

class BengioRatePyramidal : public CG_BengioRatePyramidal
{
   public:
      void update_U(RNG& rng);
      void update_Vs(RNG& rng);
      virtual void setLateralIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BengioRatePyramidalInAttrPSet* CG_inAttrPset, CG_BengioRatePyramidalOutAttrPSet* CG_outAttrPset);
      virtual ~BengioRatePyramidal();
};

#endif
