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

#ifndef NMDAReceptor_H
#define NMDAReceptor_H

#include "Lens.h"
#include "CG_NMDAReceptor.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#if RECEPTOR_NMDA == NMDAR_POINTPROCESS
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif RECEPTOR_NMDA == NMDAR_BEHABADI_2012 || \
			RECEPTOR_NMDA == NMDAR_BEHABADI_2012_MODIFIED
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#else
#define BASED_TEMPERATURE 0.0  // Celcius
#define Q10 3.0
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class NMDAReceptor : public CG_NMDAReceptor
{
   public:
      void updateNMDA(RNG& rng);
      void updateNMDADepPlasticity(RNG& rng);
      void initializeNMDA(RNG& rng);
      virtual void setPostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptorInAttrPSet* CG_inAttrPset, CG_NMDAReceptorOutAttrPSet* CG_outAttrPset);
      virtual void setPrePostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptorInAttrPSet* CG_inAttrPset, CG_NMDAReceptorOutAttrPSet* CG_outAttrPset);
      virtual ~NMDAReceptor();
  #ifdef MICRODOMAIN_CALCIUM
      virtual void setCalciumMicrodomain(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptorInAttrPSet* CG_inAttrPset, CG_NMDAReceptorOutAttrPSet* CG_outAttrPset);
      int _offset; //the offset due to the presence of different Ca2+-microdomain
  #endif
};

#endif
