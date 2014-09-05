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
