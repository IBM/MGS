#ifndef VoltageAdapter_H
#define VoltageAdapter_H

#include "Lens.h"
#include "CG_VoltageAdapter.h"
#include "rndm.h"

class VoltageAdapter : public CG_VoltageAdapter
{
   public:
      void produceInitialVoltage(RNG& rng);
      void produceVoltage(RNG& rng);
      void computeState(RNG& rng);
      virtual void setPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageAdapterInAttrPSet* CG_inAttrPset, CG_VoltageAdapterOutAttrPSet* CG_outAttrPset);
      virtual ~VoltageAdapter();
};

#endif
