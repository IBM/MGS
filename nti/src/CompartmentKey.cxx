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

#include "CompartmentKey.h"
#include "Utilities.h"
#include <cassert>
#include <iostream>
#include <float.h>

SegmentDescriptor CompartmentKey::_segmentDescriptor;

CompartmentKey::CompartmentKey() 
  : _key(0), _cptIdx(0)
{
}

CompartmentKey::CompartmentKey(CompartmentKey const & compartmentKey)
  : _key(compartmentKey._key), _cptIdx(compartmentKey._cptIdx)
{
}

CompartmentKey::CompartmentKey(double key, int cptIdx) 
  : _key(key), _cptIdx(cptIdx)
{
}

CompartmentKey::~CompartmentKey()
{
}

CompartmentKey::compare::compare() : _case(0)
{
}

CompartmentKey::compare::compare(int c) : _case(c)
{
}

bool CompartmentKey::compare::operator()(const CompartmentKey& ck0, const CompartmentKey& ck1)
{
  bool rval = false;

  unsigned int n0=_segmentDescriptor.getNeuronIndex(ck0._key);
  unsigned int n1=_segmentDescriptor.getNeuronIndex(ck1._key);

  if (n0==n1) {
    unsigned int b0=_segmentDescriptor.getBranchIndex(ck0._key);
    unsigned int b1=_segmentDescriptor.getBranchIndex(ck1._key);
 
    if (b0==b1) {
      unsigned int s0=_segmentDescriptor.getSegmentIndex(ck0._key);
      unsigned int s1=_segmentDescriptor.getSegmentIndex(ck1._key);
      
      if (_case==0 && s0==s1)
	rval=(ck0._cptIdx<ck1._cptIdx);

      else rval=(s0<s1);
      
    }
    else rval=(b0<b1);

  }
  else rval=(n0<n1);

  return rval;
}

CompartmentKey& CompartmentKey::operator=(const CompartmentKey& compartmentKey)
{
  if (this==&compartmentKey) return *this;
  _key=compartmentKey._key;
  _cptIdx=compartmentKey._cptIdx;
  return *this;
}
