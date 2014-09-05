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

#ifndef GranulePointerCompare_H
#define GranulePointerCompare_H
#include "Copyright.h"

#include "Granule.h"

// This class is used as the second parameter of  std::set, 
// if the Granule*'s should be sorted using the globalId of the Granules  
// this class is used.

class GranulePointerCompare {

   public:
      bool operator()(Granule* const & lv, Granule* const & rv) const {
	 return lv->getGlobalGranuleId() < rv->getGlobalGranuleId();
      }
};

#endif
