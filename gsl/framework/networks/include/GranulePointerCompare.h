// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
