// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CaConcentrationJunction_H
#define CaConcentrationJunction_H

#include "Lens.h"
#include "CG_CaConcentrationJunction.h"
#include "rndm.h"
#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"

class CaConcentrationJunction : public CG_CaConcentrationJunction
{
  public:
  void initializeJunction(RNG& rng);
  void predictJunction(RNG& rng);
  void correctJunction(RNG& rng);
  virtual bool checkSite(const String& CG_direction, const String& CG_component,
                         NodeDescriptor* CG_node, Edge* CG_edge,
                         VariableDescriptor* CG_variable, Constant* CG_constant,
                         CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset,
                         CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset);
  virtual bool confirmUniqueDeltaT(
      const String& CG_direction, const String& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset,
      CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset);
  virtual ~CaConcentrationJunction();
#ifdef MICRODOMAIN_CALCIUM
  virtual void createMicroDomainData(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset);
  virtual void setupCurrent2Microdomain(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset);
  std::map<int, int> _mapCurrentToMicrodomainIndex; // first 'int' = index in channelCaCurrents_microdomain
  //second 'int' = index of Ca_microdomain that this current is supposed to project to
  void updateMicrodomains(float& RHS);
  void updateMicrodomains_Ca();
#endif
  // user-defined functions
  // junction designed as 1-compartment always, there is no need for index
  dyn_var_t getVolume();
  dyn_var_t getArea();
	void printDebugHH(std::string phase="JUNCTION_CORRECT");
	private:
  static SegmentDescriptor _segmentDescriptor;
};

#endif
