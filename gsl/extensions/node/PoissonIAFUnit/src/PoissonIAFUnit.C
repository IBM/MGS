// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "PoissonIAFUnit.h"
#include "CG_PoissonIAFUnit.h"
#include "rndm.h"

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void PoissonIAFUnit::update(RNG& rng)
{
  // Produce a spike with a Poisson distribution with the given firing rate
  spike = (drandom(rng) <= (Hz / (1. / SHD.deltaT)));

  // If the simulation has reached a certain period, apply a perturbation
  if (SHD.op_perturbation && ITER == SHD.perturbationT)
    Hz = drandom(0.0, 150.0, rng);
}

PoissonIAFUnit::~PoissonIAFUnit()
{
}

