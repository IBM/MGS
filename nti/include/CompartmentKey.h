// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef COMPARTMENTKEY_H
#define COMPARTMENTKEY_H

#include <mpi.h>
#include <vector>
#include <string.h>

#include "SegmentDescriptor.h"

class CompartmentKey
{
 public:

  CompartmentKey();
  CompartmentKey(CompartmentKey const & compartmentKey);
  CompartmentKey(double key, int cptIdx=0);
  CompartmentKey& operator=(const CompartmentKey& compartmentKey);

  ~CompartmentKey();

  class compare
    {
    public:
      compare();
      compare(int c);
      bool operator()(const CompartmentKey& ck0, const CompartmentKey& ck1) const;
    private:
      int _case;
    };

  double _key;
  int _cptIdx;
  static SegmentDescriptor _segmentDescriptor;
};

#endif
