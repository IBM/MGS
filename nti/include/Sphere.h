// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

