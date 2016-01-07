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

#ifndef COMPUTEBRANCH_H
#define COMPUTEBRANCH_H

#include "Capsule.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <cassert>

class ComputeBranch
{
  public:
  ComputeBranch() : _capsules(0), _nCapsules(0), _parent(0) {}
  Capsule* _capsules;
  int _nCapsules;
  ComputeBranch* _parent;
  std::list<ComputeBranch*> _daughters; //TUAN: GPU port TODO: try to use contiguous array??
  Capsule& lastCapsule()
  {
    assert(_nCapsules > 0);
    return _capsules[_nCapsules - 1];
  }
};
#endif
