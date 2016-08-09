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

#ifndef HodgkinHuxleyVoltageJunction_H
#define HodgkinHuxleyVoltageJunction_H

#include "Lens.h"
#include "CG_HodgkinHuxleyVoltageJunction.h"
#include "rndm.h"
#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"

class HodgkinHuxleyVoltageJunction : public CG_HodgkinHuxleyVoltageJunction
{
  public:
  void initializeJunction(RNG& rng);
  void predictJunction(RNG& rng);
  void correctJunction(RNG& rng);
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
  // user-defined functions
  // junction designed as 1-compartment always, there is no need for index
  dyn_var_t getArea();
	void printDebugHH(std::string phase="JUNCTION_CORRECT");
	private:
  static SegmentDescriptor _segmentDescriptor;
};

#endif
