// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
// Created by Heraldo Memelli
// summer 2012

#ifndef BOUNDINGCUBOID_H_
#define BOUNDINGCUBOID_H_

#include "BoundingVolume.h"

class NeurogenSegment;
class NeurogenParams;

class BoundingCuboid: public BoundingVolume {
 public:
  BoundingCuboid();
   bool isOutsideVolume(NeurogenSegment* _seg);
  virtual ~BoundingCuboid();
};

#endif /* BOUNDINGCUBOID_H_ */
