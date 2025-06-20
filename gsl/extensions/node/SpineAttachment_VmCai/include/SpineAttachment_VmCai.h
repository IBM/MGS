// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SpineAttachment_VmCai_H
#define SpineAttachment_VmCai_H

#include "Mgs.h"
#include "CG_SpineAttachment_VmCai.h"
#include "rndm.h"

#include "SegmentDescriptor.h"
#include "MaxComputeOrder.h"

class SpineAttachment_VmCai : public CG_SpineAttachment_VmCai
{
  public:
  void produceInitialState(RNG& rng);
  void computeInitialState(RNG& rng);
  void produceState(RNG& rng);
  void computeState(RNG& rng);
  virtual void setCaPointers(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset,
      CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset);
  virtual void setVoltagePointers(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset,
      CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset);
  virtual void set_A_and_len(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_SpineAttachment_VmCaiInAttrPSet* CG_inAttrPset,
      CG_SpineAttachment_VmCaiOutAttrPSet* CG_outAttrPset);
  virtual ~SpineAttachment_VmCai();
  SpineAttachment_VmCai();

  private:
  dyn_var_t* _ri;
  bool _gotAssigned;
  static SegmentDescriptor _segmentDescriptor;
};

#endif
