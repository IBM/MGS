// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SegmentKeySegmentSpace.h"

SegmentKeySegmentSpace::SegmentKeySegmentSpace(std::vector<std::pair<std::string, unsigned int> > probeKey) : _probeKey(probeKey)
{
}

SegmentKeySegmentSpace::SegmentKeySegmentSpace(SegmentKeySegmentSpace& branchTypeSegmentSpace) : _probeKey(branchTypeSegmentSpace._probeKey)
{
}


bool SegmentKeySegmentSpace::isInSpace(Segment* seg)
{
  bool rval=true;
  std::vector<std::pair<std::string, unsigned int> >::iterator iter, end=_probeKey.end();
  for (iter=_probeKey.begin(); rval && iter!=end; ++iter) {
    rval = rval && 
      (_segmentDescriptor.getValue(_segmentDescriptor.getSegmentKeyData(iter->first), seg->getSegmentKey()) ==
       iter->second);
  }
  return rval;
}
 
SegmentKeySegmentSpace::~SegmentKeySegmentSpace()
{
}

SegmentSpace* SegmentKeySegmentSpace::duplicate()
{
  return new SegmentKeySegmentSpace(*this);
}
