// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef CaExtrusion_H
#define CaExtrusion_H

#include "Mgs.h"
#include "CG_CaExtrusion.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#ifndef BASED_TEMPERATURE 
#define BASED_TEMPERATURE 35.0 //Celcius
#endif

#ifndef Q10 
#define Q10 2.3 //default
#endif
// This is a simple implementation of Ca2+(cyto) extrusion
class CaExtrusion : public CG_CaExtrusion
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaExtrusionInAttrPSet* CG_inAttrPset, CG_CaExtrusionOutAttrPSet* CG_outAttrPset);
      virtual ~CaExtrusion();
	 private:
			int _cptindex;// index of the associated compartment
};

#endif
