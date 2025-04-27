#ifndef GatedThalamicUnit_H
#define GatedThalamicUnit_H

#include "Lens.h"
#include "CG_GatedThalamicUnit.h"
#include "rndm.h"
#include <fstream>

class GatedThalamicUnit : public CG_GatedThalamicUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void setIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GatedThalamicUnitInAttrPSet* CG_inAttrPset, CG_GatedThalamicUnitOutAttrPSet* CG_outAttrPset);
      virtual void outputWeights(std::ofstream& fsPH);
      virtual void inputWeight(std::ifstream& fsPH, int col);
      virtual ~GatedThalamicUnit();
};

#endif
