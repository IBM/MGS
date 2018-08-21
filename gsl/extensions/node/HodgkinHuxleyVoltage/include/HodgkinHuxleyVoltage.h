// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
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
//#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION1
#if defined(CONSIDER_MANYSPINE_EFFECT_OPTION1) || defined(CONSIDER_MANYSPINE_EFFECT_OPTION2_revised)
    virtual void updateSpineCount(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
    virtual void updateGapJunctionCount(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
#endif
    virtual void setReceptorCurrent(
        const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
    virtual void setInjectedCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
    virtual void setProximalJunction(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
    virtual bool checkSite(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
    virtual bool confirmUniqueDeltaT(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
    virtual ~HodgkinHuxleyVoltage();
#ifdef CONSIDER_DI_DV 
  virtual void add_zero_didv(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset);
#endif





















    //user-added
    //dyn_var_t getLambda(DimensionStruct* a, DimensionStruct* b, bool connectJunction=false);
    dyn_var_t getLambda(DimensionStruct* a, DimensionStruct* b, int index, bool connectJunction=false);
    dyn_var_t getLambda_parent(DimensionStruct* a, DimensionStruct* b, int index, bool connectJunction=false);
    dyn_var_t getLambda_child(DimensionStruct* a, DimensionStruct* b, int index, bool connectJunction=false);
    //dyn_var_t getLambda(DimensionStruct* a);
    dyn_var_t getLambda(DimensionStruct* a, int index);
    dyn_var_t getHalfDistance(int index);
    dyn_var_t getArea(int i);
    dyn_var_t getAij(DimensionStruct* a, DimensionStruct* b, dyn_var_t Area, bool connectJunction=false);
    dyn_var_t getAij_parent(DimensionStruct* a, DimensionStruct* b, dyn_var_t Area, bool connectJunction=false);
    dyn_var_t getAij_child(DimensionStruct* a, DimensionStruct* b, dyn_var_t Area, bool connectJunction=false);
    //unsigned getSize() {return branchData->size;}
    void printDebugHH();
    void printDebugHH(int i);
    void printDebugHHCurrent(int i);
    // - common for all compartment variables
#if MAX_COMPUTE_ORDER>0
    void forwardSolve1(RNG& rng);
    void backwardSolve1(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve1_corrector(RNG& rng);
    void backwardSolve1_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER>1
    void forwardSolve2(RNG& rng);
    void backwardSolve2(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve2_corrector(RNG& rng);
    void backwardSolve2_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER>2
    void forwardSolve3(RNG& rng);
    void backwardSolve3(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve3_corrector(RNG& rng);
    void backwardSolve3_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER>3
    void forwardSolve4(RNG& rng);
    void backwardSolve4(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve4_corrector(RNG& rng);
    void backwardSolve4_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER>4
    void forwardSolve5(RNG& rng);
    void backwardSolve5(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve5_corrector(RNG& rng);
    void backwardSolve5_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER>5
    void forwardSolve6(RNG& rng);
    void backwardSolve6(RNG& rng);
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    void forwardSolve6_corrector(RNG& rng);
    void backwardSolve6_corrector(RNG& rng);
#endif
#endif
#if MAX_COMPUTE_ORDER>6
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
#ifdef CONSIDER_DI_DV 
    dyn_var_t _zero_conductance = 0.0;
    int _tmp_index; // keep track the starting index to add zero-didv
#endif
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
    // <cpt-index, vector-all-receptors(including-SpineAttachment)-to-that-cpt>
    std::vector<std::vector<dyn_var_t> >  spineNeck_PrevPotential;
    std::vector<int> current_index; // the value indicate the current index in the above map 
    //   (keeping track of the receptors being investigated that project to 
    //the compartment index <int>
#endif
};

#endif
