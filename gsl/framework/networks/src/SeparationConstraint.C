// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SeparationConstraint.h"

SeparationConstraint::SeparationConstraint()
{
}

bool SeparationConstraint::haveCommon(const GranuleSet& granules) const
{
   GranuleSet::const_iterator it, end = granules.end(), foundElem, 
      ownedEnd = _granules.end();

   for (it = granules.begin(); it != end; ++it) {
      foundElem = _granules.find(*it);
      if (foundElem != ownedEnd) return true;      
   }   
   
   return false;
}


void SeparationConstraint::insertGranules(const GranuleSet& granules)
{
   _granules.insert(granules.begin(), granules.end());
}
