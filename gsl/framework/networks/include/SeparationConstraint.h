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
