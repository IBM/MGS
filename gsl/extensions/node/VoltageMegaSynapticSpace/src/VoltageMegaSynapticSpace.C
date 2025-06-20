// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "VoltageMegaSynapticSpace.h"
#include "CG_VoltageMegaSynapticSpace.h"
#include "rndm.h"
#include "CG_VoltageAdapter.h"
#include "Coordinates.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"

#define DEBUG
#define THRESHOLD -30

void VoltageMegaSynapticSpace::produceInitialVoltage(RNG& rng) 
{
   _numInputs = Vm.size();
   _BinWidth = 10.0 ; // msec  - spikes within thin windows are counted TODO - modulable
   //_timeSpikesAtInput = new float[_numInputs]();
   for (int _i=0; _i< _numInputs; _i++)
   {
      _timeSpikesAtInput.push_back(0.0-_BinWidth);
      _spikeAtInputIsOn.push_back(0);
   }
#ifdef DEBUG
   std::cerr << "MegaSS with globalIdx = " 
      << this->getGlobalIndex()
      << ", receives " << _numInputs << "neurons" << std::endl;
   //for (auto& n : Vm)
   //   std::cerr << "AAA ";
   std::cerr << std::endl;
   std::cerr << std::endl;
#endif
}

void VoltageMegaSynapticSpace::produceVoltage(RNG& rng) 
{
}

void VoltageMegaSynapticSpace::computeState(RNG& rng) 
{
   //aggregate all ShallowArray<dyn_var_t*> Vm
   //to produce LFP
   LFP = -70.0; // resting (mV) 
   //std::for_each(Vm.begin(), Vm.end(), [&])
   for (auto& n : Vm)
      LFP +=  *n;
   LFP /= _numInputs;
#define TStep (*(getSharedMembers().deltaT))  // msec
   float currentTime = (getSimulation().getIteration() * TStep);
   numSpikes = 0;
   int _i = 0;
   for (auto& n : Vm)
   {
      if (*n > THRESHOLD)
      {
         if (_spikeAtInputIsOn[_i] == false)
         {
            _timeSpikesAtInput[_i] = currentTime;
            _spikeAtInputIsOn[_i] = true;
         }
      }
      else if (_spikeAtInputIsOn[_i] == true)
      {
         _spikeAtInputIsOn[_i] = false;
      }
      if (_timeSpikesAtInput[_i] >= currentTime - _BinWidth)
      {
         numSpikes++;
      }
      _i++;
   }
   // for (auto& n : Vm)
   //    if (*n > THRESHOLD)
   //       numSpikes++;
//#ifdef DEBUG
//   //if (LFP > -50)
//   {
//      std::cerr << "LFP: " << LFP << " numSpikes = " << numSpikes << std::endl;
//      //for (auto& n : Vm)
//      //   std::cerr << " " <<  *n ;
//   }
//#endif
}

VoltageMegaSynapticSpace::~VoltageMegaSynapticSpace() 
{
   //if (_timeSpikesAtInput)
   //   delete[] _timeSpikesAtInput;
}

