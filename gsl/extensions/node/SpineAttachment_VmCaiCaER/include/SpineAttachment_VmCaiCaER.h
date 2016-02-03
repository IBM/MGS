#ifndef SpineAttachment_VmCaiCaER_H
#define SpineAttachment_VmCaiCaER_H

#include "Lens.h"
#include "CG_SpineAttachment_VmCaiCaER.h"
#include "rndm.h"

class SpineAttachment_VmCaiCaER : public CG_SpineAttachment_VmCaiCaER
{
   public:
      void produceInitialState(RNG& rng);
      void produceState(RNG& rng);
      void computeState(RNG& rng);
      virtual void setVoltagePointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset);
      virtual void setCaPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset);
      virtual void setCaERPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset);
      virtual void set_A_and_len(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineAttachment_VmCaiCaERInAttrPSet* CG_inAttrPset, CG_SpineAttachment_VmCaiCaEROutAttrPSet* CG_outAttrPset);
      virtual ~SpineAttachment_VmCaiCaER();
};

#endif
