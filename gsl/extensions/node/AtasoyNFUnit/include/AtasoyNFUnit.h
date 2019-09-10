#ifndef AtasoyNFUnit_H
#define AtasoyNFUnit_H

#include "Lens.h"
#include "CG_AtasoyNFUnit.h"
#include "rndm.h"

class AtasoyNFUnit : public CG_AtasoyNFUnit
{
   public:
      void initialize(RNG& rng);
      void diffusion(RNG& rng);
      void reaction(RNG& rng);
      virtual bool checkForConnection(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_AtasoyNFUnitInAttrPSet* CG_inAttrPset, CG_AtasoyNFUnitOutAttrPSet* CG_outAttrPset);
      virtual ~AtasoyNFUnit();
};

#endif
