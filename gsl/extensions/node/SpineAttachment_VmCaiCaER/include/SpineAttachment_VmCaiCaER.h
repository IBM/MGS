// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef SpineAttachment_VmCaiCaER_H
#define SpineAttachment_VmCaiCaER_H

#include "Mgs.h"
#include "CG_SpineAttachment_VmCaiCaER.h"
#include "rndm.h"
#include "SegmentDescriptor.h"

#include "MaxComputeOrder.h"

class SpineAttachment_VmCaiCaER : public CG_SpineAttachment_VmCaiCaER
{
  public:
  void produceInitialState(RNG& rng);
  void computeInitialState(RNG& rng);
  void produceState(RNG& rng);
  void computeState(RNG& rng);
  virtual void setVoltagePointers(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset,
      CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset);
  virtual void setCaPointers(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset,
      CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset);
  virtual void setCaERPointers(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset,
      CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset);
  virtual void set_A_and_len(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset,
      CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset);
  virtual ~SpineAttachment_VmCaiCaER();
  SpineAttachment_VmCaiCaER();

  private:
  dyn_var_t* _ri;
  bool _gotAssigned;
  static SegmentDescriptor _segmentDescriptor;
};

#endif
