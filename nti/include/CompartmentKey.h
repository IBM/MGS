// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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
      bool operator()(const CompartmentKey& ck0, const CompartmentKey& ck1);
    private:
      int _case;
    };

  double _key;
  int _cptIdx;
  static SegmentDescriptor _segmentDescriptor;
};

#endif
