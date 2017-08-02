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
