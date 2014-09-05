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

#ifndef NMDAReceptor_H
#define NMDAReceptor_H

#include "Lens.h"
#include "CG_NMDAReceptor.h"
#include "rndm.h"

class NMDAReceptor : public CG_NMDAReceptor
{
   public:
      void updateNMDA(RNG& rng);
      void updateNMDADepPlasticity(RNG& rng);
      void initializeNMDA(RNG& rng);
      virtual void setPostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptorInAttrPSet* CG_inAttrPset, CG_NMDAReceptorOutAttrPSet* CG_outAttrPset);
      float sigmoid(float alpha, float beta);
      virtual ~NMDAReceptor();
};

#endif
