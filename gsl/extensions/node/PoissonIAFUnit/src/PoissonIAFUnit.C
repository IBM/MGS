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

