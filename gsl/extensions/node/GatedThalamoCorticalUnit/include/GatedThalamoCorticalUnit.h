#ifndef GatedThalamoCorticalUnit_H
#define GatedThalamoCorticalUnit_H

#include "Lens.h"
#include "CG_GatedThalamoCorticalUnit.h"
#include "rndm.h"
#include <fstream>

class GatedThalamoCorticalUnit : public CG_GatedThalamoCorticalUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void whiten(RNG& rng);
      virtual void setIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GatedThalamoCorticalUnitInAttrPSet* CG_inAttrPset, CG_GatedThalamoCorticalUnitOutAttrPSet* CG_outAttrPset);
      virtual void outputWeights(std::ofstream& fsPH);
      virtual void inputWeight(std::ifstream& fsPH, int col);
      virtual void getLateralCovInputs(std::ofstream& fsLN);
      virtual void setLateralWhitInputs(std::vector<double>* latWhitInputs);
      virtual ~GatedThalamoCorticalUnit();
};

#endif
