#ifndef Connexon_VmCaiCaER_H
#define Connexon_VmCaiCaER_H

#include "Lens.h"
#include "CG_Connexon_VmCaiCaER.h"
#include "rndm.h"

class Connexon_VmCaiCaER : public CG_Connexon_VmCaiCaER
{
   public:
      void produceInitialState(RNG& rng);
      void produceState(RNG& rng);
      void computeState(RNG& rng);
      virtual void setVoltagePointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_Connexon_VmCaiCaERInAttrPSet* CG_inAttrPset, CG_Connexon_VmCaiCaEROutAttrPSet* CG_outAttrPset);
      virtual void setCaPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_Connexon_VmCaiCaERInAttrPSet* CG_inAttrPset, CG_Connexon_VmCaiCaEROutAttrPSet* CG_outAttrPset);
      virtual void setCaERPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_Connexon_VmCaiCaERInAttrPSet* CG_inAttrPset, CG_Connexon_VmCaiCaEROutAttrPSet* CG_outAttrPset);
      virtual ~Connexon_VmCaiCaER();
};

#endif
