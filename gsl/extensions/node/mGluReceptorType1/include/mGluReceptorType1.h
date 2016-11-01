#ifndef mGluReceptorType1_H
#define mGluReceptorType1_H

#include "Lens.h"
#include "CG_mGluReceptorType1.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

class mGluReceptorType1 : public CG_mGluReceptorType1
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void setPostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_mGluReceptorType1InAttrPSet* CG_inAttrPset, CG_mGluReceptorType1OutAttrPSet* CG_outAttrPset);
#ifdef KEEP_PAIR_PRE_POST
      //do nothing
#else
      virtual void setPreIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_mGluReceptorType1InAttrPSet* CG_inAttrPset, CG_mGluReceptorType1OutAttrPSet* CG_outAttrPset);
#endif
      virtual ~mGluReceptorType1();
   private:
      dyn_var_t sigmoid(dyn_var_t alpha, dyn_var_t beta);
};

#endif
