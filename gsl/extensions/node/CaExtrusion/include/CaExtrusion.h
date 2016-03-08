#ifndef CaExtrusion_H
#define CaExtrusion_H

#include "Lens.h"
#include "CG_CaExtrusion.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

// This is a simple implementation of Ca2+(cyto) extrusion
class CaExtrusion : public CG_CaExtrusion
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaExtrusionInAttrPSet* CG_inAttrPset, CG_CaExtrusionOutAttrPSet* CG_outAttrPset);
      virtual ~CaExtrusion();
	 private:
			int _cptindex;// index of the associated compartment
};

#endif
