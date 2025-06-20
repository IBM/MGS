// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SeparationConstraint_H
#define SeparationConstraint_H
#include "Copyright.h"

#include "GranuleSet.h"

class SeparationConstraint
{
   public:
      SeparationConstraint();
      const GranuleSet& getGranules() {
	 return _granules;
      }

      bool haveCommon(const GranuleSet& granules) const;
      void insertGranules(const GranuleSet& granules);
      
   private:
      GranuleSet _granules;
};

#endif
