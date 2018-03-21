#include "Lens.h"
#include "VoltageMegaSynapticSpace.h"
#include "CG_VoltageMegaSynapticSpace.h"
#include "rndm.h"
#include "CG_VoltageAdapter.h"
#include "Coordinates.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"

void VoltageMegaSynapticSpace::produceInitialVoltage(RNG& rng) 
{
   _numInputs = Vm.size();
}

void VoltageMegaSynapticSpace::produceVoltage(RNG& rng) 
{
}

void VoltageMegaSynapticSpace::computeState(RNG& rng) 
{
   //aggregate all ShallowArray<dyn_var_t*> Vm
   //to produce LFP
   LFP = 0.0; 
   //std::for_each(Vm.begin(), Vm.end(), [&])
   for (auto& n : Vm)
      LFP +=  *n;
   LFP /= _numInputs;
}

VoltageMegaSynapticSpace::~VoltageMegaSynapticSpace() 
{
}

