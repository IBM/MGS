// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef GABAAReceptor_H
#define GABAAReceptor_H

#include "Lens.h"
#include "CG_GABAAReceptor.h"
#include "rndm.h"
#include "MaxComputeOrder.h"

#if RECEPTOR_GABA == GABAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif RECEPTOR_GABA == GABAR_POINTPROCESS
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#endif
class GABAAReceptor : public CG_GABAAReceptor
{
   public:
      void updateGABAA(RNG& rng);
      void initializeGABAA(RNG& rng);
      virtual void setPostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GABAAReceptorInAttrPSet* CG_inAttrPset, CG_GABAAReceptorOutAttrPSet* CG_outAttrPset);
      virtual ~GABAAReceptor();
};

#endif
