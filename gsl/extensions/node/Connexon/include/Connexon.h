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

#ifndef Connexon_H
#define Connexon_H

#include "Lens.h"
#include "CG_Connexon.h"
#include "rndm.h"

class Connexon : public CG_Connexon
{
   public:
      void produceInitialVoltage(RNG& rng);
      void produceVoltage(RNG& rng);
      void computeState(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ConnexonInAttrPSet* CG_inAttrPset, CG_ConnexonOutAttrPSet* CG_outAttrPset);
      virtual ~Connexon();
};

#endif
