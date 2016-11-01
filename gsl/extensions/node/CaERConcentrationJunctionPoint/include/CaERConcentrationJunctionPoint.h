#ifndef CaERConcentrationJunctionPoint_H
#define CaERConcentrationJunctionPoint_H

#include "Lens.h"
#include "CG_CaERConcentrationJunctionPoint.h"
#include "rndm.h"

class CaERConcentrationJunctionPoint : public CG_CaERConcentrationJunctionPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceCaConcentration(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationJunctionPointInAttrPSet* CG_inAttrPset, CG_CaERConcentrationJunctionPointOutAttrPSet* CG_outAttrPset);
      virtual ~CaERConcentrationJunctionPoint();
};

#endif
