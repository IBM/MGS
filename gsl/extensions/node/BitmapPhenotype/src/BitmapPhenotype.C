// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "BitmapPhenotype.h"
#include "CG_BitmapPhenotype.h"
#include "rndm.h"

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void BitmapPhenotype::initialize(RNG& rng) 
{
  std::vector<int> coords;
  getNodeCoords(coords);
  r = coords[0];
  c = coords[1];
}

void BitmapPhenotype::update(RNG& rng) 
{
  int phase = ITER % SHD.period;
  if (phase == 0) {
    int idx = (SHD.row + r) * SHD.imgCols[SHD.imageNbr] + (SHD.col + c);
    x = double(image[SHD.imageNbr][idx])/256.0;
    x0 += getSharedMembers().betaX0 * (x - x0);
    x -= x0;
  }
}

BitmapPhenotype::~BitmapPhenotype() 
{
}
