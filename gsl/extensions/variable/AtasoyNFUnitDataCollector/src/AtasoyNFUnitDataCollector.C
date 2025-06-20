// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "AtasoyNFUnitDataCollector.h"
#include "CG_AtasoyNFUnitDataCollector.h"
#include <memory>

void AtasoyNFUnitDataCollector::initialize(RNG& rng) 
{
}

void AtasoyNFUnitDataCollector::finalize(RNG& rng) 
{
}

void AtasoyNFUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
}

AtasoyNFUnitDataCollector::AtasoyNFUnitDataCollector() 
   : CG_AtasoyNFUnitDataCollector()
{
}

AtasoyNFUnitDataCollector::~AtasoyNFUnitDataCollector() 
{
}

void AtasoyNFUnitDataCollector::duplicate(std::unique_ptr<AtasoyNFUnitDataCollector>&& dup) const
{
   dup.reset(new AtasoyNFUnitDataCollector(*this));
}

void AtasoyNFUnitDataCollector::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new AtasoyNFUnitDataCollector(*this));
}

void AtasoyNFUnitDataCollector::duplicate(std::unique_ptr<CG_AtasoyNFUnitDataCollector>&& dup) const
{
   dup.reset(new AtasoyNFUnitDataCollector(*this));
}

