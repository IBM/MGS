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

#ifndef BOUNDINGSURFACEMESH_H_
#define BOUNDINGSURFACEMESH_H_

#include "BoundingVolume.h"
#include <string.h>
#include <string>

class NeurogenSegment;
class NeurogenParams;

class BoundingSurfaceMesh: public BoundingVolume {
 public:
  BoundingSurfaceMesh(std::string filename);
  bool isOutsideVolume(NeurogenSegment* _seg);
  virtual ~BoundingSurfaceMesh();

  double* _hx;
  double* _hy;
  double* _hz;
  int _npts;

 private:
  int _ntriangles;
  int* _A;
  int* _B;
  int* _C;
  double* _norms;
  double* _distPtsSqrd;
  double* _distTrgSqrd;
  double _meanX;
  double _meanY;
  double _meanZ;
  double _minDistSqrd;
  bool boundless;
};

#endif /* BOUNDINGSURFACEMESH_H_ */
