// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef SPHERE_H
#define SPHERE_H

#include <mpi.h>

#define N_SPHERE_DATA 6

struct Sphere
{
  double _coords[3];
  double _radius;
  double _key;
  double _dist2Soma;
};
#endif

