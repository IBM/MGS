// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef mGluReceptorType1_H
#define mGluReceptorType1_H

#include "Mgs.h"
#include "CG_mGluReceptorType1.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

class mGluReceptorType1 : public CG_mGluReceptorType1
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void setPostIndex(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_mGluReceptorType1InAttrPSet* CG_inAttrPset, CG_mGluReceptorType1OutAttrPSet* CG_outAttrPset);
      virtual void setPrePostIndex(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_mGluReceptorType1InAttrPSet* CG_inAttrPset, CG_mGluReceptorType1OutAttrPSet* CG_outAttrPset);
#ifdef KEEP_PAIR_PRE_POST
      //do nothing
#else
      virtual void setPreIndex(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_mGluReceptorType1InAttrPSet* CG_inAttrPset, CG_mGluReceptorType1OutAttrPSet* CG_outAttrPset);
#endif
      virtual ~mGluReceptorType1();
   private:
      dyn_var_t sigmoid(dyn_var_t alpha, dyn_var_t beta);
};

#endif
