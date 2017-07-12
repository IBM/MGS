#ifndef ToneUnit_H
#define ToneUnit_H

#include "Lens.h"
#include "CG_ToneUnit.h"
#include "rndm.h"

class ToneUnit : public CG_ToneUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ToneUnitInAttrPSet* CG_inAttrPset, CG_ToneUnitOutAttrPSet* CG_outAttrPset);
      virtual ~ToneUnit();
};

#endif
