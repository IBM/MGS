/*
 * BoundingCuboid.cpp
 *
 *  Created on: Jul 31, 2012
 *      Author: heraldo
 */

#include "BoundingCuboid.h"
#include "NeurogenParams.h"
#include "NeurogenSegment.h"

BoundingCuboid::BoundingCuboid()
{
}

bool BoundingCuboid::isOutsideVolume(NeurogenSegment* _seg)
{
  bool rval=false;
  if (fabs(_seg->getX() - _seg->getParams()->startX) > _seg->getParams()->width/2.0)
    rval=true;
  else if (fabs(_seg->getY() - _seg->getParams()->startY) > _seg->getParams()->height/2.0)
    rval=true;
  else if (fabs(_seg->getZ() - _seg->getParams()->startZ) > _seg->getParams()->depth/2.0)
    rval=true;
  return rval;
}

BoundingCuboid::~BoundingCuboid()
{
}
