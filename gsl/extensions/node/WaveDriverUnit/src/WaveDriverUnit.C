// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "WaveDriverUnit.h"
#include "CG_WaveDriverUnit.h"
#include "rndm.h"

#define SHD getSharedMembers()

void WaveDriverUnit::initialize(RNG& rng) 
{
  // Default values
  wave = 0.0; // just in case something uses it before it is updated
}

void WaveDriverUnit::update(RNG& rng) 
{
  // First if set by the user, modulate the parameters either with
  // Brownian noise or a weight Brownian noise.
  if (op_modulateHz)
    {
      if (op_modulateHzWeight)
        Hz += ((HzA + (HzB * Hz)) * gaussian(rng)) * HzModRate;
      else
        Hz += gaussian(rng) * HzModRate;
      if (Hz > HzMax)
        Hz = HzMax;
      if (Hz < HzMin)
        Hz = HzMin;
    }
  if (op_modulatePhase)
    {
      if (op_modulatePhaseWeight)
        phase += ((phaseA + (phaseB * phase)) * gaussian(rng)) * phaseModRate;
      else
        phase += gaussian(rng) * phaseModRate;
      if (phase > phaseMax)
        phase = phaseMax;
      if (phase < phaseMin)
        phase = phaseMin;
    }
  if (op_modulateAmplitude)
    {
      if (op_modulateAmplitudeWeight)
        amplitude += ((amplitudeA + (amplitudeB * amplitude)) * gaussian(rng)) * amplitudeModRate;
      else
        amplitude += gaussian(rng) * amplitudeModRate;
      if (amplitude > amplitudeMax)
        amplitude = amplitudeMax;
      if (amplitude < amplitudeMin)
        amplitude = amplitudeMin;
    }

  // Now update the wave with a sine wave ...
  wave = amplitude
    * fabs(
       (
        pow(sin(
            (
             (
              2 * M_PI * Hz
              )
             * getSimulation().getIteration() * SHD.deltaT
             )
            + phase
            ), 9.0)
        + 0.0 // offset
        )
       / 2.0
       );
}

void WaveDriverUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_WaveDriverUnitInAttrPSet* CG_inAttrPset, CG_WaveDriverUnitOutAttrPSet* CG_outAttrPset) 
{
}

WaveDriverUnit::~WaveDriverUnit() 
{
}

