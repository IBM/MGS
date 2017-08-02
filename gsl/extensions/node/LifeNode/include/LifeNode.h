// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef LifeNode_H
#define LifeNode_H

#include "Lens.h"
#include "CG_LifeNode.h"
#include "rndm.h"

class LifeNode : public CG_LifeNode
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void copy(RNG& rng);
      virtual ~LifeNode();
};

#endif
