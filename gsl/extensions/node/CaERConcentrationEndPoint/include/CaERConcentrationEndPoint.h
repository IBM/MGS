#ifndef CaERConcentrationEndPoint_H
#define CaERConcentrationEndPoint_H

#include "Lens.h"
#include "CG_CaERConcentrationEndPoint.h"
#include "rndm.h"

class CaERConcentrationEndPoint : public CG_CaERConcentrationEndPoint
{
   public:
      void produceInitialState(RNG& rng);
      void produceSolvedCaConcentration(RNG& rng);
      void produceFinishedCaConcentration(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationEndPointInAttrPSet* CG_inAttrPset, CG_CaERConcentrationEndPointOutAttrPSet* CG_outAttrPset);
      virtual ~CaERConcentrationEndPoint();
};

#endif
