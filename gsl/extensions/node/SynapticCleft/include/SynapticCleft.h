#ifndef SynapticCleft_H
#define SynapticCleft_H
/* =================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-07-18-2017

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

=================================================================
*/

#include "CG_SynapticCleft.h"
#include "Lens.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

//#define DEBUG

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
#ifdef DEBUG
  std::string data_received;
#endif
};

#endif
