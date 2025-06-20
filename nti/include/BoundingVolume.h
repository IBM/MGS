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

#ifndef BOUNDINGVOLUME_H_
#define BOUNDINGVOLUME_H_

class NeurogenSegment;

class BoundingVolume
{
 public:
  BoundingVolume(){}

  virtual bool isOutsideVolume(NeurogenSegment*)=0;
  virtual ~BoundingVolume() {}

};


#endif /* BOUNDINGVOLUME_H_ */
