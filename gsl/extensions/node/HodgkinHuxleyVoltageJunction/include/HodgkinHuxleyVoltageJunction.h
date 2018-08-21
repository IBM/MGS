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

#ifndef HodgkinHuxleyVoltageJunction_H
#define HodgkinHuxleyVoltageJunction_H

#include "Lens.h"
#include "CG_HodgkinHuxleyVoltageJunction.h"
#include "rndm.h"
#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"

#define DEBUG_COMPARTMENT

class HodgkinHuxleyVoltageJunction : public CG_HodgkinHuxleyVoltageJunction
{
  public:
  void initializeJunction(RNG& rng);
  void predictJunction(RNG& rng);
  void correctJunction(RNG& rng);
//#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION1
#if defined(CONSIDER_MANYSPINE_EFFECT_OPTION1) || defined(CONSIDER_MANYSPINE_EFFECT_OPTION2_revised)
  virtual void updateSpineCount(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset);
  virtual void updateGapJunctionCount(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset);
#endif
  virtual bool checkSite(
      const String& CG_direction, const String& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset,
      CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset);
  virtual bool confirmUniqueDeltaT(
      const String& CG_direction, const String& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset,
      CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset);
  virtual ~HodgkinHuxleyVoltageJunction();
  virtual void add_zero_didv(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset);
  virtual void select_current_stream(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset);
#ifdef CONSIDER_EFFECT_LARGE_CHANGE_CURRENT_STIMULATE
  virtual void update_stim_reference(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset);
  int _count_timer = 0;
#endif

  // user-defined functions
  // junction designed as 1-compartment always, there is no need for index
  dyn_var_t getArea();
  void printDebugHH(std::string phase="JUNCTION_CORRECT");
  private:
  static SegmentDescriptor _segmentDescriptor;
  dyn_var_t _zero_conductance = 0.0;
#ifdef INTRINSIC_NOISE_TO_NEURON
  dyn_var_t delay_until_adjust_noise;  // in msec (until noise_factor_current_injection is reevaluated)
  dyn_var_t checked_time; // msec (time-point reference to calculate delay_until_adjust_noise)
#endif
};

#endif
