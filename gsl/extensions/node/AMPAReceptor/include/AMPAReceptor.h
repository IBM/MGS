// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
