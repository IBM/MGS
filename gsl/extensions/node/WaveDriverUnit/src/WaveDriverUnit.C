// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
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

#define ITER getSimulation().getIteration()
#define SHD getSharedMembers()

void WaveDriverUnit::initialize(RNG& rng) 
{
}

void WaveDriverUnit::update(RNG& rng) 
{
  if (ITER <= 1)
    {
      Hz /= 2.0;
      amplitude *= 2.0;      
    }
  if (!SHD.op_constant)
    {
      // First if set by the user, modulate the parameters either with
      // Brownian noise or a weight Brownian noise.
      if (SHD.op_modulateHz)
        {
          if (SHD.op_modulateHzWeight)
            Hz += ((HzA + (HzB * Hz)) * gaussian(rng)) * HzModRate;
          else
            Hz += gaussian(rng) * HzModRate;
          if (Hz > HzMax)
            Hz = HzMax;
          if (Hz < HzMin)
            Hz = HzMin;
        }
      if (SHD.op_modulatePhase)
        {
          if (SHD.op_modulatePhaseWeight)
            phase += ((phaseA + (phaseB * phase)) * gaussian(rng)) * phaseModRate;
          else
            phase += gaussian(rng) * phaseModRate;
          if (phase > phaseMax)
            phase = phaseMax;
          if (phase < phaseMin)
            phase = phaseMin;
        }
      if (SHD.op_modulateAmplitude)
        {
          if (SHD.op_modulateAmplitudeWeight)
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
}

void WaveDriverUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_WaveDriverUnitInAttrPSet* CG_inAttrPset, CG_WaveDriverUnitOutAttrPSet* CG_outAttrPset) 
{
}

WaveDriverUnit::~WaveDriverUnit() 
{
}

