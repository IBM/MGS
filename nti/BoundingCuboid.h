// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-2012
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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
