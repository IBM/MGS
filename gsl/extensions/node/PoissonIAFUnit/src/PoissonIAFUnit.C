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
#include "PoissonIAFUnit.h"
#include "CG_PoissonIAFUnit.h"
#include "rndm.h"

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void PoissonIAFUnit::initialize(RNG& rng)
{
}

void PoissonIAFUnit::update(RNG& rng)
{
  // Produce a spike with a Poisson distribution with the given firing rate
  spike = (drandom(rng) <= (Hz / (1. / SHD.deltaT)));

  // If the simulation has reached a certain period, apply a perturbation
  if (ITER == 5000000)
    Hz = drandom(0.0, 150.0, rng);
}

PoissonIAFUnit::~PoissonIAFUnit()
{
}

