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
