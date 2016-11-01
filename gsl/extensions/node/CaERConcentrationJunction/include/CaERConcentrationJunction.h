#ifndef CaERConcentrationJunction_H
#define CaERConcentrationJunction_H

#include "Lens.h"
#include "CG_CaERConcentrationJunction.h"
#include "rndm.h"
#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"

class CaERConcentrationJunction : public CG_CaERConcentrationJunction
{
  public:
  void initializeJunction(RNG& rng);
  void predictJunction(RNG& rng);
  void correctJunction(RNG& rng);
  virtual bool checkSite(
      const String& CG_direction, const String& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_CaERConcentrationJunctionInAttrPSet* CG_inAttrPset,
      CG_CaERConcentrationJunctionOutAttrPSet* CG_outAttrPset);
  virtual bool confirmUniqueDeltaT(
      const String& CG_direction, const String& CG_component,
      NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
      Constant* CG_constant,
      CG_CaERConcentrationJunctionInAttrPSet* CG_inAttrPset,
      CG_CaERConcentrationJunctionOutAttrPSet* CG_outAttrPset);
  virtual ~CaERConcentrationJunction();
  // user-defined functions
  // junction designed as 1-compartment always, there is no need for index
  dyn_var_t getVolume();
  dyn_var_t getArea();
	private:
  static SegmentDescriptor _segmentDescriptor;
};

#endif
