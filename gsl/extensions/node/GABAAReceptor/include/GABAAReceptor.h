// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================
// 
// (C) Copyright 2018 New Jersey Institute of Technology. All rights reserved.
// 
// =============================================================================

#ifndef GABAAReceptor_H
#define GABAAReceptor_H

#include "Lens.h"
#include "CG_GABAAReceptor.h"
#include "rndm.h"
#include "MaxComputeOrder.h"

#if RECEPTOR_GABAA == GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif RECEPTOR_GABAA == GABAAR_POINTPROCESS
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif RECEPTOR_GABAA == GABAAR_MULTIPLEPARAMS
#define BASED_TEMPERATURE 25.0  // arbitrary, Q10 = 1
#define Q10 1.0
#endif


#ifndef Q10 
#define Q10 3.0 //default
#endif
class GABAAReceptor : public CG_GABAAReceptor
{
   public:
      void updateGABAA(RNG& rng);
      void initializeGABAA(RNG& rng);
      virtual void setPostIndex(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GABAAReceptorInAttrPSet* CG_inAttrPset, CG_GABAAReceptorOutAttrPSet* CG_outAttrPset);
      virtual void setPrePostIndex(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GABAAReceptorInAttrPSet* CG_inAttrPset, CG_GABAAReceptorOutAttrPSet* CG_outAttrPset);
      virtual ~GABAAReceptor();
};

#endif
