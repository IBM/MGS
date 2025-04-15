// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef IP3ConcentrationJunction_H
#define IP3ConcentrationJunction_H

#include "Lens.h"
#include "CG_IP3ConcentrationJunction.h"
#include "rndm.h"
#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"

class IP3ConcentrationJunction : public CG_IP3ConcentrationJunction
{
  public:
  void initializeJunction(RNG& rng);
  void predictJunction(RNG& rng);
  void correctJunction(RNG& rng);
  virtual bool checkSite(const String& CG_direction, const String& CG_component,
                         NodeDescriptor* CG_node, Edge* CG_edge,
                         VariableDescriptor* CG_variable, Constant* CG_constant,
                         CG_IP3ConcentrationJunctionInAttrPSet* CG_inAttrPset,
                         CG_IP3ConcentrationJunctionOutAttrPSet* CG_outAttrPset);
  virtual bool confirmUniqueDeltaT(
      const String& CG_direction, const String& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_IP3ConcentrationJunctionInAttrPSet* CG_inAttrPset,
      CG_IP3ConcentrationJunctionOutAttrPSet* CG_outAttrPset);
  virtual ~IP3ConcentrationJunction();
  // user-defined functions
  // junction designed as 1-compartment always, there is no need for index
  dyn_var_t getVolume();
  dyn_var_t getArea();
	void printDebugHH(std::string phase="JUNCTION_CORRECT");
	private:
  static SegmentDescriptor _segmentDescriptor;
};

#endif
