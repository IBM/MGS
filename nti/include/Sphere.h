// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef SPHERE_H
#define SPHERE_H

#include "../../nti/include/MaxComputeOrder.h"
#include <mpi.h>

//TUAN
//key_size_t : be careful to modify this accordingly if we change key_size_t
//IMPORTANT: The value here represent the number of 'double' element
//  which is important for mapping to MPI_DOUBLE for sending data
#define N_SPHERE_DATA 6

struct Sphere
{
  double _coords[3];
  double _radius;
  //TUAN: as key size can change, it is suggested to move it to the end of the struct
  key_size_t _key;
  double _dist2Soma; // along-the-branch-distance to the soma
};
#endif

