// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef VoltageMegaSynapticSpace_H
#define VoltageMegaSynapticSpace_H

#include "Mgs.h"
#include "CG_VoltageMegaSynapticSpace.h"
#include "rndm.h"

class VoltageMegaSynapticSpace : public CG_VoltageMegaSynapticSpace
{
   public:
      void produceInitialVoltage(RNG& rng);
      void produceVoltage(RNG& rng);
      void computeState(RNG& rng);
      virtual ~VoltageMegaSynapticSpace();
   private:
      int _numInputs;
      //float* _timeSpikesAtInput;
      std::vector<float> _timeSpikesAtInput;
      std::vector<bool> _spikeAtInputIsOn; //keep track if the spike at a given neuron is off or not
      float _BinWidth;
};

#endif
