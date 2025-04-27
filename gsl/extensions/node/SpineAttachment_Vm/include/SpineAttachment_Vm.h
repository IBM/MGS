// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SpineAttachment_Vm_H
#define SpineAttachment_Vm_H

#include "Lens.h"
#include "CG_SpineAttachment_Vm.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

class SpineAttachment_Vm : public CG_SpineAttachment_Vm
{
  public:
  void produceInitialState(RNG& rng);
  void computeInitialState(RNG& rng);
  void produceState(RNG& rng);
  void computeState(RNG& rng);
  virtual void setVoltagePointers(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_SpineAttachment_VmInAttrPSet* CG_inAttrPset,
      CG_SpineAttachment_VmOutAttrPSet* CG_outAttrPset);
  virtual void set_A_and_len(const CustomString& CG_direction,
                             const CustomString& CG_component,
                             NodeDescriptor* CG_node, Edge* CG_edge,
                             VariableDescriptor* CG_variable,
                             Constant* CG_constant,
                             CG_SpineAttachment_VmInAttrPSet* CG_inAttrPset,
                             CG_SpineAttachment_VmOutAttrPSet* CG_outAttrPset);
  virtual ~SpineAttachment_Vm();
  SpineAttachment_Vm();
  private:
  dyn_var_t* _ri;
  bool _gotAssigned;
};

#endif
