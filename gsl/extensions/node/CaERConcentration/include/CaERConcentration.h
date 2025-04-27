// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CaERConcentration_H
#define CaERConcentration_H

#include "Lens.h"
#include "CG_CaERConcentration.h"
#include "SegmentDescriptor.h"
#include "MaxComputeOrder.h"
#include "rndm.h"

class CaERConcentration : public CG_CaERConcentration
{
  public:
  void initializeCompartmentData(RNG& rng);
  void solve(RNG& rng);
  void finish(RNG& rng);
  //  virtual void setReceptorCaCurrent(
  //      const CustomString& CG_direction, const CustomString& CG_component,
  //      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor*
  //      CG_variable,
  //      Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset,
  //      CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
  virtual void setInjectedCaCurrent(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset,
      CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
  virtual void setProximalJunction(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset,
      CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
  virtual void setTargetAttachCaConcentration(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
  virtual bool checkSite(const CustomString& CG_direction, const CustomString& CG_component,
                         NodeDescriptor* CG_node, Edge* CG_edge,
                         VariableDescriptor* CG_variable, Constant* CG_constant,
                         CG_CaERConcentrationInAttrPSet* CG_inAttrPset,
                         CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
  virtual bool confirmUniqueDeltaT(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset,
      CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
  virtual ~CaERConcentration();

  // added
  // - specific to one compartment variable
  dyn_var_t getLambda(DimensionStruct* a, DimensionStruct* b);
  dyn_var_t getAij(DimensionStruct* a, DimensionStruct* b, dyn_var_t volume);
  dyn_var_t getVolume(int i);
  dyn_var_t getArea(int i);
  void printDebugHH();
  void printDebugHH(int i);

// - common for all compartment variables
#if MAX_COMPUTE_ORDER > 0
  void forwardSolve1(RNG& rng);
  void backwardSolve1(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER > 1
  void forwardSolve2(RNG& rng);
  void backwardSolve2(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER > 2
  void forwardSolve3(RNG& rng);
  void backwardSolve3(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER > 3
  void forwardSolve4(RNG& rng);
  void backwardSolve4(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER > 4
  void forwardSolve5(RNG& rng);
  void backwardSolve5(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER > 5
  void forwardSolve6(RNG& rng);
  void backwardSolve6(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER > 6
  void forwardSolve7(RNG& rng);
  void backwardSolve7(RNG& rng);
#endif
  private:
  void doForwardSolve();
  void doBackwardSolve();
  static SegmentDescriptor _segmentDescriptor;
};

#endif
