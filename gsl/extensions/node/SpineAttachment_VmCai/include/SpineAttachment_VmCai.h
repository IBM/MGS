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

#ifndef SpineAttachment_VmCai_H
#define SpineAttachment_VmCai_H

#include "Lens.h"
#include "CG_SpineAttachment_VmCai.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

class SpineAttachment_VmCai : public CG_SpineAttachment_VmCai
{
   public:
      void produceInitialState(RNG& rng);
      void produceState(RNG& rng);
      void computeState(RNG& rng);
      virtual void setCaPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset);
      virtual void setVoltagePointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset);
      virtual void set_A_and_len(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset);
      virtual ~SpineAttachment_VmCai();
			dyn_var_t * _ri;
};

#endif
