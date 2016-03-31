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

#ifndef HodgkinHuxleyVoltage_H
#define HodgkinHuxleyVoltage_H

#include "Lens.h"
#include "CG_HodgkinHuxleyVoltage.h"
#include "SegmentDescriptor.h"
#include "MaxComputeOrder.h"

#include "rndm.h"

class HodgkinHuxleyVoltage : public CG_HodgkinHuxleyVoltage
{
   public:
      void initializeCompartmentData(RNG& rng);
      void solve(RNG& rng);
      void finish(RNG& rng);
      virtual void setReceptorCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
      virtual void setInjectedCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
      virtual void setProximalJunction(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
      virtual bool checkSite(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
      virtual bool confirmUniqueDeltaT(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
      virtual ~HodgkinHuxleyVoltage();
	  
	  //user-added
      dyn_var_t getLambda(DimensionStruct* a, DimensionStruct* b);
      dyn_var_t getArea(int i);
      dyn_var_t getAij(DimensionStruct* a, DimensionStruct* b, dyn_var_t Area);
      //unsigned getSize() {return branchData->size;}
	
#if MAX_COMPUTE_ORDER>0
      void forwardSolve1(RNG& rng);
      void backwardSolve1(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>1
      void forwardSolve2(RNG& rng);
      void backwardSolve2(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>2
      void forwardSolve3(RNG& rng);
      void backwardSolve3(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>3
      void forwardSolve4(RNG& rng);
      void backwardSolve4(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>4
      void forwardSolve5(RNG& rng);
      void backwardSolve5(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>5
      void forwardSolve6(RNG& rng);
      void backwardSolve6(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>6
      void forwardSolve7(RNG& rng);
      void backwardSolve7(RNG& rng);
#endif

   private:
      void doForwardSolve();
      void doBackwardSolve();
};

#endif
