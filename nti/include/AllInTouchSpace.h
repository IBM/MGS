// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ALLINTOUCHSPACE_H
#define ALLITTOUCHSPACE_H
#include "TouchSpace.h"
#include "MaxComputeOrder.h"

#include <mpi.h>

class AllInTouchSpace : public TouchSpace
{
 public:
  AllInTouchSpace();
  AllInTouchSpace(AllInTouchSpace&);
  ~AllInTouchSpace();
  bool isInSpace(key_size_t segKey1);
  bool areInSpace(key_size_t segKey1, key_size_t segKey2);
  TouchSpace* duplicate();
};
 
#endif

