// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GranuleSet_H
#define GranuleSet_H
#include "Copyright.h"

#include "GranulePointerCompare.h"
#include <set>

typedef std::set<Granule*, GranulePointerCompare> GranuleSet;

#endif
