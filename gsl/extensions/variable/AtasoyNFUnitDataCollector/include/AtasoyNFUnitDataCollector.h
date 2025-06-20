// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef AtasoyNFUnitDataCollector_H
#define AtasoyNFUnitDataCollector_H

#include "Mgs.h"
#include "CG_AtasoyNFUnitDataCollector.h"
#include <memory>

class AtasoyNFUnitDataCollector : public CG_AtasoyNFUnitDataCollector
{
   public:
      void initialize(RNG& rng);
      void finalize(RNG& rng);
      virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
      AtasoyNFUnitDataCollector();
      virtual ~AtasoyNFUnitDataCollector();
      virtual void duplicate(std::unique_ptr<AtasoyNFUnitDataCollector>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_AtasoyNFUnitDataCollector>&& dup) const;
};

#endif
