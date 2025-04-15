// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef AMPAReceptor_H
#define AMPAReceptor_H

#include "Lens.h"
#include "CG_AMPAReceptor.h"
#include "rndm.h"
#include "MaxComputeOrder.h"

#if RECEPTOR_AMPA == AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif RECEPTOR_AMPA == AMPAR_POINTPROCESS
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#else
#define BASED_TEMPERATURE 0.0  // Celcius
#define Q10 3.0
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class AMPAReceptor : public CG_AMPAReceptor
{
   public:
      void updateAMPA(RNG& rng);
      void initializeAMPA(RNG& rng);
      virtual void setPostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_AMPAReceptorInAttrPSet* CG_inAttrPset, CG_AMPAReceptorOutAttrPSet* CG_outAttrPset);
      virtual void setPrePostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_AMPAReceptorInAttrPSet* CG_inAttrPset, CG_AMPAReceptorOutAttrPSet* CG_outAttrPset);
      virtual ~AMPAReceptor();
};

#endif
