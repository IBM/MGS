// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef IP3Concentration_H
#define IP3Concentration_H

#include "Lens.h"
#include "CG_IP3Concentration.h"
#include "SegmentDescriptor.h"
#include "MaxComputeOrder.h"

#include "rndm.h"

class IP3Concentration : public CG_IP3Concentration
{
  public:
    void initializeCompartmentData(RNG& rng);
    void solve(RNG& rng);
    void finish(RNG& rng);
    virtual void setReceptorIP3Current(
        const String& CG_direction, const String& CG_component,
        NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
        Constant* CG_constant, CG_IP3ConcentrationInAttrPSet* CG_inAttrPset,
        CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset);
    virtual void setInjectedIP3Current(
        const String& CG_direction, const String& CG_component,
        NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
        Constant* CG_constant, CG_IP3ConcentrationInAttrPSet* CG_inAttrPset,
        CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset);
    virtual void setProximalJunction(
        const String& CG_direction, const String& CG_component,
        NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
        Constant* CG_constant, CG_IP3ConcentrationInAttrPSet* CG_inAttrPset,
        CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset);
      virtual void setTargetAttachIP3Concentration(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IP3ConcentrationInAttrPSet* CG_inAttrPset, CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset);
    virtual bool checkSite(const String& CG_direction, const String& CG_component,
        NodeDescriptor* CG_node, Edge* CG_edge,
        VariableDescriptor* CG_variable, Constant* CG_constant,
        CG_IP3ConcentrationInAttrPSet* CG_inAttrPset,
        CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset);
    virtual bool confirmUniqueDeltaT(
        const String& CG_direction, const String& CG_component,
        NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
        Constant* CG_constant, CG_IP3ConcentrationInAttrPSet* CG_inAttrPset,
        CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset);
    virtual ~IP3Concentration();

    // user-added
    // - specific to one compartment variable
    //dyn_var_t getLambda(DimensionStruct* a, DimensionStruct* b, bool connectJunction=false);
    dyn_var_t getLambda(DimensionStruct* a, DimensionStruct* b, int index, bool connectJunction=false);
    //dyn_var_t getLambda(DimensionStruct* a);
    dyn_var_t getLambda(DimensionStruct* a, int index);
    dyn_var_t getHalfDistance(int index);
    dyn_var_t getArea(int i);
    dyn_var_t getAij(DimensionStruct* a, DimensionStruct* b, dyn_var_t volume, bool connectJunction=false);
    dyn_var_t getVolume(int i);
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
