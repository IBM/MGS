// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-2012
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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
