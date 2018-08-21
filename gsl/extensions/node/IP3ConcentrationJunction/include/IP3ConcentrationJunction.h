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
