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

#ifndef COMPUTEBRANCH_H
#define COMPUTEBRANCH_H

#include "Capsule.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <cassert>

#include "MaxComputeOrder.h"

class ComputeBranch
{
  public:
  ComputeBranch() : _capsules(0), _nCapsules(0), _parent(0) 
#ifdef IDEA1
  , _configuredCompartment(false)
#endif
  {};
  Capsule& lastCapsule();
  //DATA
  Capsule* _capsules;
  int _nCapsules;

  ComputeBranch* _parent;
  std::list<ComputeBranch*> _daughters; //TUAN: GPU port TODO: try to use contiguous array??
  dyn_var_t getSurfaceArea();

#ifdef IDEA1
  std::pair<float, float> 
      _numCapsulesEachSideForBranchPoint;  // keep info. to
  // determine how many capsule is reserved for a branchpoint
  std::vector<int>  _cptSizesForBranch;  // keep info. to
  bool _configuredCompartment;
  //// store the #caps/compartment on that ComputeBranch
  //int getNumCompartments(std::vector<int>& cptsizes_in_branch,
  //                       bool& isDistalEndSeeImplicitBranchingPoint);
#endif
};
#endif
