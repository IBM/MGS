// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CaConcentration_H
#define CaConcentration_H

#include "Lens.h"
#include "CG_CaConcentration.h"
#include "SegmentDescriptor.h"
#include "MaxComputeOrder.h"

#include "rndm.h"

class CaConcentration : public CG_CaConcentration
{
  public:
  void initializeCompartmentData(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    //void forwardSolve0(RNG& rng); //We don't need this, just use solve()
    //void backwardSolve0(RNG& rng); //we don't need this, just use solve()
    void backwardSolve0_corrector(RNG& rng);
    void forwardSolve0_corrector(RNG& rng);
    void doForwardSolve_corrector();
//#else
//    void solve(RNG& rng);
#endif
  void solve(RNG& rng);
  void finish(RNG& rng);
  virtual void setReceptorCaCurrent(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset,
      CG_CaConcentrationOutAttrPSet* CG_outAttrPset);
  virtual void setInjectedCaCurrent(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset,
      CG_CaConcentrationOutAttrPSet* CG_outAttrPset);
  virtual void setProximalJunction(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset,
      CG_CaConcentrationOutAttrPSet* CG_outAttrPset);
  virtual void setTargetAttachCaConcentration(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset,
      CG_CaConcentrationOutAttrPSet* CG_outAttrPset);
  virtual bool checkSite(const CustomString& CG_direction, const CustomString& CG_component,
                         NodeDescriptor* CG_node, Edge* CG_edge,
                         VariableDescriptor* CG_variable, Constant* CG_constant,
                         CG_CaConcentrationInAttrPSet* CG_inAttrPset,
                         CG_CaConcentrationOutAttrPSet* CG_outAttrPset);
  virtual bool confirmUniqueDeltaT(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset,
      CG_CaConcentrationOutAttrPSet* CG_outAttrPset);
  virtual ~CaConcentration();
#ifdef MICRODOMAIN_CALCIUM
  virtual void createMicroDomainData(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset,
      CG_CaConcentrationOutAttrPSet* CG_outAttrPset);
  virtual void setupCurrent2Microdomain(
      const CustomString& CG_direction, const CustomString& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset,
      CG_CaConcentrationOutAttrPSet* CG_outAttrPset);
  std::map<int, int>
      _mapCurrentToMicrodomainIndex;  // first 'int' = index in
                                      // channelCaCurrents_microdomain
// second 'int' = index of Ca_microdomain that this current is supposed to project to
  std::map<int, int> _mapFluxToMicrodomainIndex; // first 'int' = index in channelCaFluxes_microdomain
  virtual void setupFlux2Microdomain(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset, CG_CaConcentrationOutAttrPSet* CG_outAttrPset);
  void updateMicrodomains();
  void updateMicrodomains_Ca();
#endif

  // user-added
  // - specific to one compartment variable
  // dyn_var_t getLambda(DimensionStruct* a, DimensionStruct* b, bool
  // connectJunction=false);
  dyn_var_t getLambda(DimensionStruct* a, DimensionStruct* b, int index,
                      bool connectJunction = false);
  dyn_var_t getLambda_parent(DimensionStruct* a, DimensionStruct* b, int index,
                      bool connectJunction = false);
  dyn_var_t getLambda_child(DimensionStruct* a, DimensionStruct* b, int index,
                      bool connectJunction = false);
  // dyn_var_t getLambda(DimensionStruct* a);
  dyn_var_t getLambda(DimensionStruct* a, int index);
  dyn_var_t getHalfDistance(int index);
  dyn_var_t getArea(int i);
  dyn_var_t getAij(DimensionStruct* a, DimensionStruct* b, dyn_var_t volume,
                   bool connectJunction = false);
  dyn_var_t getAij_parent(DimensionStruct* a, DimensionStruct* b, dyn_var_t volume,
                   bool connectJunction = false);
  dyn_var_t getAij_child(DimensionStruct* a, DimensionStruct* b, dyn_var_t volume,
                   bool connectJunction = false);
  dyn_var_t getVolume(int i);
  void printDebugHH();
  void printDebugHH(int i);

// - common for all compartment variables
#if MAX_COMPUTE_ORDER > 0
  void forwardSolve1(RNG& rng);
  void backwardSolve1(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve1_corrector(RNG& rng);
    void backwardSolve1_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER > 1
  void forwardSolve2(RNG& rng);
  void backwardSolve2(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve2_corrector(RNG& rng);
    void backwardSolve2_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER > 2
  void forwardSolve3(RNG& rng);
  void backwardSolve3(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve3_corrector(RNG& rng);
    void backwardSolve3_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER > 3
  void forwardSolve4(RNG& rng);
  void backwardSolve4(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve4_corrector(RNG& rng);
    void backwardSolve4_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER > 4
  void forwardSolve5(RNG& rng);
  void backwardSolve5(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve5_corrector(RNG& rng);
    void backwardSolve5_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER > 5
  void forwardSolve6(RNG& rng);
  void backwardSolve6(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve6_corrector(RNG& rng);
    void backwardSolve6_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER > 6
  void forwardSolve7(RNG& rng);
  void backwardSolve7(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve7_corrector(RNG& rng);
    void backwardSolve7_corrector(RNG& rng);
#endif
#endif
  private:
  void doForwardSolve();
  void doBackwardSolve();
  static SegmentDescriptor _segmentDescriptor;
};

#endif
