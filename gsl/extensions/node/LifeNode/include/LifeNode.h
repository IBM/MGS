// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LifeNode_H
#define LifeNode_H

#include "Lens.h"
#include "CG_LifeNode.h"
#include "rndm.h"
#include "NumberUtils.h"

#define F_SIGMOID 0 
#define F_ReLU    1
#define F_TANH 2
class LifeNode : public CG_LifeNode
{
   public:
      ///TUAN NOTE: we cann't make them '__global__' if we use static data member '_container'
      //  This is ok, as we don't put data member inside class
      //CUDA_CALLABLE void initialize(RNG& rng);
      //CUDA_CALLABLE void update(RNG& rng);
      //CUDA_CALLABLE void copy(RNG& rng);
      void initialize(RNG& rng);
      void update(RNG& rng);
      void updateWeight(RNG& rng);
      void copy(RNG& rng);
      virtual ~LifeNode();
};

#endif
