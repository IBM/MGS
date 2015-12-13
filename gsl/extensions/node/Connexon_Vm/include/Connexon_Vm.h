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

#ifndef Connexon_Vm_H
#define Connexon_Vm_H

#include "Lens.h"
#include "CG_Connexon_Vm.h"
#include "rndm.h"

class Connexon_Vm : public CG_Connexon_Vm
{
   public:
      void produceInitialVoltage(RNG& rng);
      void produceVoltage(RNG& rng);
      void computeState(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_Connexon_VmInAttrPSet* CG_inAttrPset, CG_Connexon_VmOutAttrPSet* CG_outAttrPset);
      virtual ~Connexon_Vm();
};

#endif
