// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef SingleChannelRYRCompCategory_H
#define SingleChannelRYRCompCategory_H

#include "Mgs.h"
#include "CG_SingleChannelRYRCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class SingleChannelRYRCompCategory : public CG_SingleChannelRYRCompCategory,
                                     public CountableModel
{
  public:
  SingleChannelRYRCompCategory(Simulation& sim, const std::string& modelName,
                               const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void setupChannel(RNG& rng);
  ~SingleChannelRYRCompCategory();
  void count();

  public:
  // dyn_var_t channelDensity; // [1/um^2] - the # of RyR per 1um^2
  // int numStates;
  // int * vOpenStates; // vector telling which state is open state
  // = 1 (open), = 0 (closed)
  // dyn_var_t ** matChannelRateConstant;
  /*        int ** stateFromTo; //reduced-form matrix of transition rate
     constant
          int ** indxK; // tracking original index in reduced-form matrix
          int ** StateSpace;
          int maxNumNeighbors;
                                  */
};

#endif
