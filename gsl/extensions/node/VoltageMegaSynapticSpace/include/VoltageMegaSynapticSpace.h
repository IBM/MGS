#ifndef VoltageMegaSynapticSpace_H
#define VoltageMegaSynapticSpace_H

#include "Lens.h"
#include "CG_VoltageMegaSynapticSpace.h"
#include "rndm.h"

class VoltageMegaSynapticSpace : public CG_VoltageMegaSynapticSpace
{
   public:
      void produceInitialVoltage(RNG& rng);
      void produceVoltage(RNG& rng);
      void computeState(RNG& rng);
      bool isInRange(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageMegaSynapticSpaceInAttrPSet* CG_inAttrPset, CG_VoltageMegaSynapticSpaceOutAttrPSet* CG_outAttrPset);
      virtual ~VoltageMegaSynapticSpace();
   private:
      int _numInputs;
};

#endif
