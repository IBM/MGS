// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef EpileptorNode_H
#define EpileptorNode_H

#include "Mgs.h"
#include "CG_EpileptorNode.h"
#include "rndm.h"
#include<map>

#define Jirsa_et_al_2014
//#define Proix_et_al_2014
//#define Proix_et_al_2018

class EpileptorNode : public CG_EpileptorNode
{
   public:
      void initialize(RNG& rng);
      void updateDeltas(RNG& rng);
      void update(RNG& rng);
      virtual ~EpileptorNode();
      std::map<std::pair<int, int>, float> connectionMap;
};

#endif
