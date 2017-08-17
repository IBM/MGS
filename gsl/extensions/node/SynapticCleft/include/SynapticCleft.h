#ifndef SynapticCleft_H
#define SynapticCleft_H

#include "CG_SynapticCleft.h"
#include "Lens.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

class SynapticCleft : public CG_SynapticCleft
{
  public:
  void produceInitialState(RNG& rng);
  void produceState(RNG& rng);
  virtual void setPointers(const String& CG_direction,
                           const String& CG_component, NodeDescriptor* CG_node,
                           Edge* CG_edge, VariableDescriptor* CG_variable,
                           Constant* CG_constant,
                           CG_SynapticCleftInAttrPSet* CG_inAttrPset,
                           CG_SynapticCleftOutAttrPSet* CG_outAttrPset);
  virtual ~SynapticCleft();
  private:
  float _timeLastSpike = 0; //from the last spike of pre-synaptic
  bool _reset = true;
};

#endif
