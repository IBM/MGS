#ifndef PumpSERCA_H
#define PumpSERCA_H

#include "Lens.h"
#include "CG_PumpSERCA.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#ifndef BASED_TEMPERATURE 
#define BASED_TEMPERATURE 35.0 //Celcius
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class PumpSERCA : public CG_PumpSERCA
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual void setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_PumpSERCAInAttrPSet* CG_inAttrPset, CG_PumpSERCAOutAttrPSet* CG_outAttrPset);
      virtual ~PumpSERCA();
	 private:
			int _cptindex;// index of the associated compartment
};

#endif
