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
