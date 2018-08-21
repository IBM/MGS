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
